/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "userbuffers.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>

namespace tensorrt_llm::kernels::ub
{
using namespace tensorrt_llm::runtime::ub;
#define MAX_THREADS 1024
#define TIMEOUT 200000000000ull

template <typename DType, int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw(int const op, int const flagoffset, int const firstrank, int const myrank,
        int const gpustep, size_t const lineoffset, int const numlines, void** commbuff, int const handleridx)
{
    __shared__ int4* userptr[RANKS];
    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = (*reduceidptr) + 1;
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;

        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &(myptr[targetgpu]);
        userptr[threadIdx.x] = reinterpret_cast<int4*>(commbuff[targetgpu + handleridx]);
        clock_t s = clock64();
        while (*flag < reduce_id)
        {
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
        }
        reduce_id++;
    }
    __syncthreads();

    int warp = blockIdx.x + (threadIdx.x >> 5);
    int dest[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++)
        dest[i] = (i + myrank + warp) & (RANKS - 1);

    __syncthreads();
    for (int line = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x); line < numlines;
         line += blockDim.x * gridDim.x * RANKS)
    {
        int4 val[RANKS];

#pragma unroll
        for (int i = 0; i < RANKS; i++)
        {
            val[i] = userptr[dest[i]][lineoffset + line];
        }

        int4 sum = val[0];
        DType* s = reinterpret_cast<DType*>(&sum);

#pragma unroll
        for (int i = 1; i < RANKS; i++)
        {
            DType* x = reinterpret_cast<DType*>(&val[i]);
#pragma unroll
            for (int j = 0; j < 8; j++)
                s[j] += x[j];
        }
#pragma unroll
        for (int i = 0; i < RANKS; i++)
        {
            userptr[dest[i]][lineoffset + line] = sum;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &myptr[targetgpu];
        clock_t s = clock64();
        while (*flag < reduce_id)
        {
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Volta,Hopper)

template <typename DType, int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr(int const op, int const flagoffset, int const firstrank, int const myrank,
        int const gpustep, size_t const lineoffset, int const numlines, void** commbuff, int const handleridx)
{
    __shared__ int4* userptr[RANKS];
    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = (*reduceidptr) + 1;
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;

        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &(myptr[targetgpu]);
        userptr[threadIdx.x] = reinterpret_cast<int4*>(commbuff[targetgpu + handleridx]);
        clock_t s = clock64();
        while (*flag < reduce_id)
        {
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
        }
        reduce_id++;
    }
    __syncthreads();

    int warp = blockIdx.x + (threadIdx.x >> 5);
    int dest[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++)
        dest[i] = (i + myrank + warp) & (RANKS - 1);

    __syncthreads();
    for (int line = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x); line < numlines;
         line += blockDim.x * gridDim.x * RANKS)
    {
        int4 val[RANKS];

#pragma unroll
        for (int i = 0; i < RANKS; i++)
        {
            val[i] = userptr[dest[i]][lineoffset + line];
        }

        int4 sum = val[0];
        DType* s = reinterpret_cast<DType*>(&sum);

#pragma unroll
        for (int i = 1; i < RANKS; i++)
        {
            DType* x = reinterpret_cast<DType*>(&val[i]);
#pragma unroll
            for (int j = 0; j < 8; j++)
                s[j] += x[j];
        }

        userptr[myrank][lineoffset + line] = sum;
    }
#ifdef ALLREDUCEONLYRS
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
    return;
#endif
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &myptr[targetgpu];
        clock_t s = clock64();
        while (*flag < reduce_id)
        {
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
        }
    }

    int skipmy = 0;
#pragma unroll
    for (int i = 0; i < RANKS; i++)
    {
        int dst = (i + warp + myrank) & (RANKS - 1);
        if (dst == myrank)
        {
            skipmy++;
            continue;
        }
        dest[i - skipmy] = dst;
    }
    __syncthreads();

    for (int line = threadIdx.x + blockDim.x * RANKS * blockIdx.x; line < numlines;
         line += blockDim.x * gridDim.x * RANKS)
    {
        int4 val[RANKS - 1];

#pragma unroll
        for (int i = 0; i < RANKS - 1; i++)
        {
            val[i] = userptr[dest[i]][lineoffset + line + blockDim.x * dest[i]];
        }

#pragma unroll
        for (int i = 0; i < RANKS - 1; i++)
        {
            userptr[myrank][lineoffset + line + blockDim.x * dest[i]] = val[i];
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Ampere)

#define ATOMIC_CONSUMER(chunk)                                                                                         \
    if (counters)                                                                                                      \
    {                                                                                                                  \
        if (threadIdx.x == 0 && blockIdx.x == 0)                                                                       \
        {                                                                                                              \
            int old_val;                                                                                               \
            while (0 != (old_val = atomicCAS(((unsigned int*) counters) + chunk, 0, 0)))                               \
            {                                                                                                          \
            }                                                                                                          \
            ((unsigned int*) counters)[chunk] = 1;                                                                     \
            asm volatile("fence.sc.gpu;\n");                                                                           \
        }                                                                                                              \
        if (blockIdx.x == 0)                                                                                           \
            __syncthreads();                                                                                           \
    }

#define ATOMIC_PRODUCER(chunk)                                                                                         \
    if (counters)                                                                                                      \
    {                                                                                                                  \
        ((unsigned int*) counters)[chunk] = 0;                                                                         \
        asm volatile("fence.sc.gpu;\n");                                                                               \
    }

#if __CUDA_ARCH__ >= 900
template <typename ValType, typename PtrType>
__device__ __forceinline__ void MULTIMEM_ST(ValType val, PtrType ptr)
{
    asm volatile(
        "multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
}

template <typename ValType, typename PtrType>
__device__ __forceinline__ void MULTIMEM_ST2(ValType& val, PtrType ptr)
{
    asm volatile("multimem.st.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y) : "memory");
}

template <typename DType, typename ValType, typename PtrType>
__device__ __forceinline__ void MULTIMEM_LD(ValType& val, PtrType ptr)
{
    if constexpr (std::is_same_v<DType, half>)
    {
        asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
            : "l"(ptr)
            : "memory");
    }
#ifdef ENABLE_BF16
    if constexpr (std::is_same_v<DType, __nv_bfloat16>)
    {
        asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
            : "l"(ptr)
            : "memory");
    }
#endif
}

// All MC kernels here
template <typename DType, int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc(int const op, int const flagoffset,
    int const firstrank, int const myrank, int const gpustep, size_t const lineoffset, int const numlines,
    void** commbuff, int const handleridx, float4* mc_ptr)
{
    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = (*reduceidptr) + 1;
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;

        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &(myptr[targetgpu]);
#ifdef UB_TIMEOUT_ENABLED
        clock_t s = clock64();
#endif
        while (*flag < reduce_id)
        {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
#endif
        }
        reduce_id++;
    }
    __syncthreads();
#define UNROLL_MC 8
    int const loop_step0 = blockDim.x * gridDim.x * RANKS;
    int const loop_step = loop_step0 * UNROLL_MC;
    int const start_elem = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x);
    int const end_elem = max(start_elem, numlines);
    int const aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
    int const end_aligned = start_elem + aligned_elem;

    for (int line = start_elem; line < end_aligned; line += loop_step)
    {
        uint4 val[UNROLL_MC];
#pragma unroll
        for (int i = 0; i < UNROLL_MC; i++)
            MULTIMEM_LD<DType>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));
#pragma unroll
        for (int i = 0; i < UNROLL_MC; i++)
            MULTIMEM_ST(val[i], mc_ptr + (lineoffset + line + i * loop_step0));
    }
    for (int line = end_aligned; line < end_elem; line += loop_step0)
    {
        uint4 val;
        MULTIMEM_LD<DType>(val, mc_ptr + (lineoffset + line));
        MULTIMEM_ST(val, mc_ptr + (lineoffset + line));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &myptr[targetgpu];
#ifdef UB_TIMEOUT_ENABLED
        clock_t s = clock64();
#endif
        while (*flag < reduce_id)
        {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > 2ull * TIMEOUT)
            {
                printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
#endif
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Hopper) MC

#else
template <typename DType, int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc(int const op, int const flagoffset,
    int const firstrank, int const myrank, int const gpustep, size_t const lineoffset, int const numlines,
    void** commbuff, int const handleridx, float4* mc_ptr)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

#endif

#define callranks(x)                                                                                                   \
    if (ar_nvsize == x)                                                                                                \
    {                                                                                                                  \
        int arg1 = op - MAX_OPS,                                                                                       \
            arg2 = REG0_OFFSET(comm) - (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,   \
            arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step;                                                      \
        size_t arg6 = offset / 8;                                                                                      \
        int arg7 = elements / 8;                                                                                       \
        void** arg8 = (void**) (comm->gpu_ptrs);                                                                       \
        int arg9 = handler * comm->nvsize;                                                                             \
        void* kernelArgs[]                                                                                             \
            = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2), reinterpret_cast<void*>(&arg3),         \
                reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5), reinterpret_cast<void*>(&arg6),        \
                reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8), reinterpret_cast<void*>(&arg9)};       \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(&cfg,                                                                      \
            (void*) (comm->use_rr_kernel ? userbuffers_fp16_sum_inplace_gpu_rr<DType, x>                               \
                                         : userbuffers_fp16_sum_inplace_gpu_rw<DType, x>),                             \
            kernelArgs));                                                                                              \
    }

#define callranksMC(x)                                                                                                 \
    if (ar_nvsize == x)                                                                                                \
    {                                                                                                                  \
        int arg1 = op - MAX_OPS,                                                                                       \
            arg2 = REG0_OFFSET(comm) - (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,   \
            arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step;                                                      \
        size_t arg6 = offset / 8;                                                                                      \
        int arg7 = elements / 8;                                                                                       \
        void** arg8 = (void**) (comm->gpu_ptrs);                                                                       \
        int arg9 = handler * comm->nvsize;                                                                             \
        void* arg10 = comm->mc_ptr[handler];                                                                           \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10)};                                          \
        TLLM_CUDA_CHECK(                                                                                               \
            cudaLaunchKernelExC(&cfg, (void*) (userbuffers_fp16_sum_inplace_gpu_mc<DType, x>), kernelArgs));           \
    }

#define SETUP_LAUNCH_CONFIG(sms, threads, stream)                                                                      \
    cudaLaunchConfig_t cfg = {sms, threads, 0, stream, NULL, 0};                                                       \
    cudaLaunchAttribute attribute_ub[3];                                                                               \
    attribute_ub[2].id = cudaLaunchAttributeClusterDimension;                                                          \
    attribute_ub[2].val.clusterDim.x = sms % comm->cga_size == 0 ? comm->cga_size : 1;                                 \
    attribute_ub[2].val.clusterDim.y = 1;                                                                              \
    attribute_ub[2].val.clusterDim.z = 1;                                                                              \
    attribute_ub[0].id = cudaLaunchAttributeCooperative;                                                               \
    attribute_ub[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                           \
    attribute_ub[1].val.programmaticStreamSerializationAllowed = comm->pdl_launch;                                     \
    cfg.attrs = attribute_ub;                                                                                          \
    cfg.numAttrs = comm->sm_arch >= 9 ? 3 : 1;

template <typename DType>
__inline__ __device__ float compute_rmsnorm2(float val, float s_variance, DType const* gamma, DType const* beta, int i)
{
    float ret = val * s_variance * (float) (gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + (float) (beta[i]);
    }
    return ret;
}

#define shard_tokens(ntokens, nranks, myrank)                                                                          \
    int first_token = 0, my_tokens;                                                                                    \
    {                                                                                                                  \
        int remapped_rank = myrank;                                                                                    \
        my_tokens = ntokens / nranks;                                                                                  \
        int extra_tokens = ntokens % nranks;                                                                           \
        first_token = remapped_rank * my_tokens;                                                                       \
        first_token += remapped_rank < extra_tokens ? remapped_rank : extra_tokens;                                    \
        if (remapped_rank < extra_tokens)                                                                              \
            my_tokens++;                                                                                               \
    }

#if __CUDA_ARCH__ >= 900

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, float2* mc_ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    float const sf = 1.f / (*scale);
    __shared__ float s_variance;
    int hidden_dim = blockDim.x * UNROLL_NLINES * sizeof(int4) / sizeof(DType);

    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = (*reduceidptr) + 1;
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;

        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &(myptr[targetgpu]);
#ifdef UB_TIMEOUT_ENABLED
        clock_t s = clock64();
#endif
        while (*flag < reduce_id)
        {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
#endif
        }
        reduce_id++;
    }
    __syncthreads();

    int const loop_step0 = blockDim.x;
    int const loop_step = loop_step0 * UNROLL_NLINES * gridDim.x;
    int const start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL_NLINES;
    int const end_elem = max(start_elem, numlines);

    for (int line = start_elem; line < end_elem; line += loop_step)
    {
        uint4 val[UNROLL_NLINES];
        DType* x = reinterpret_cast<DType*>(&val[0]);
#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            MULTIMEM_LD<DType>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

        if (residual_in != nullptr)
        {
#pragma unroll
            for (int i = 0; i < UNROLL_NLINES; i++)
            {
                uint4 resval = residual_in[res_offset + line + i * loop_step0];
                DType* y = reinterpret_cast<DType*>(&resval);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    x[i * 8 + j] += y[j];
                residual_out[res_offset + line + i * loop_step0] = val[i];
            }
        }

        float local_var_sum = 0.0f;
        for (int j = 0; j < UNROLL_NLINES * sizeof(int4) / sizeof(DType); j++)
            local_var_sum += (float) (x[j]) * (float) (x[j]);

        float packed[1] = {local_var_sum};
        blockReduceSumV2<float, 1>(packed);
        float variance = packed[0];

        if (threadIdx.x == 0)
        {
            variance = (variance / hidden_dim); // Var[x] = E[x²]
            s_variance = rsqrtf(variance + eps);
        }
        __syncthreads();

        int i = 0;
        uint2 valout;
        __nv_fp8_e4m3* y = reinterpret_cast<__nv_fp8_e4m3*>(&valout);
#pragma unroll
        for (int g = 0; g < UNROLL_NLINES; g++)
        {
#pragma unroll
            for (int j = 0; j < sizeof(int4) / sizeof(DType); j++)
            {
                y[j] = cuda_cast<__nv_fp8_e4m3>(sf
                    * compute_rmsnorm2<DType>((float) x[i], s_variance, gamma, beta,
                        (threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(DType) + j));
                i++;
            }
            MULTIMEM_ST2(valout, mc_ptr_out + (out_lineoffset + line + g * loop_step0));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &myptr[targetgpu];
#ifdef UB_TIMEOUT_ENABLED
        clock_t s = clock64();
#endif
        while (*flag < reduce_id)
        {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > 2ull * TIMEOUT)
            {
                printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
#endif
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // quant kernel fp16->fp8 twoshot

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, uint2* mc_ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    float const sf = 1.f / (*scale);
    __shared__ float s_variance;
    int hidden_dim = blockDim.x * UNROLL_NLINES * sizeof(int4) / sizeof(DType);

    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = (*reduceidptr) + 1;
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;

        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &(myptr[targetgpu]);
#ifdef UB_TIMEOUT_ENABLED
        clock_t s = clock64();
#endif
        while (*flag < reduce_id)
        {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > TIMEOUT)
            {
                printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
#endif
        }
    }
    __syncthreads();

    int const loop_step0 = blockDim.x;
    int const loop_step = loop_step0 * UNROLL_NLINES * gridDim.x;
    int const start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL_NLINES;
    int const end_elem = max(start_elem, numlines);

    for (int line = start_elem; line < end_elem; line += loop_step)
    {
        uint4 val[UNROLL_NLINES];
        DType* x = reinterpret_cast<DType*>(&val[0]);
#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            MULTIMEM_LD<DType>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

        if (residual_in != nullptr)
        {
#pragma unroll
            for (int i = 0; i < UNROLL_NLINES; i++)
            {
                uint4 resval = residual_in[res_offset + line + i * loop_step0];
                DType* y = reinterpret_cast<DType*>(&resval);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    x[i * 8 + j] += y[j];
                residual_out[res_offset + line + i * loop_step0] = val[i];
            }
        }

        float local_var_sum = 0.0f;
        for (int j = 0; j < UNROLL_NLINES * sizeof(int4) / sizeof(DType); j++)
            local_var_sum += (float) (x[j]) * (float) (x[j]);

        float packed[1] = {local_var_sum};
        blockReduceSumV2<float, 1>(packed);
        float variance = packed[0];

        if (threadIdx.x == 0)
        {
            variance = (variance / hidden_dim); // Var[x] = E[x²]
            s_variance = rsqrtf(variance + eps);
        }
        __syncthreads();

        int i = 0;
        uint2 valout;
        __nv_fp8_e4m3* y = reinterpret_cast<__nv_fp8_e4m3*>(&valout);

#pragma unroll
        for (int g = 0; g < UNROLL_NLINES; g++)
        {
#pragma unroll
            for (int j = 0; j < sizeof(int4) / sizeof(DType); j++)
            {
                y[j] = cuda_cast<__nv_fp8_e4m3>(sf
                    * compute_rmsnorm2<DType>((float) x[i], s_variance, gamma, beta,
                        (threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(DType) + j));
                i++;
            }
            mc_ptr_out[out_lineoffset + line + g * loop_step0] = valout;
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // quant kernel fp16->fp8 oneshot

inline __device__ void load128(uint4 const* ptr, uint4& val)
{
    uint64_t* v = (uint64_t*) &val;
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v[0]), "=l"(v[1]) : "l"(ptr));
}

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot_lamport(
    int const myrank, uint4* ptr_in, int const numlines, int* reduceidptr, uint4* buff_ptr, float4* mc_ptr,
    DType const* beta, DType const* gamma, float const eps, int const RANKS, uint2* ptr_out,
    size_t const out_lineoffset, float const* scale, uint4* residual_in, uint4* residual_out)
{
    __shared__ int reduce_id;

    if (threadIdx.x == 0)
    {
        reduce_id = (*reduceidptr) + 1;
        if (blockIdx.x == 0)
            *reduceidptr = reduce_id;
    }
    __syncthreads();

    int const loop_step0 = blockDim.x;
    int const loop_step = loop_step0 * UNROLL_NLINES * gridDim.x;
    int const start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL_NLINES;
    int const end_elem = max(start_elem, numlines);

    for (int line = start_elem; line < end_elem; line += loop_step)
    {
        uint4 val[UNROLL_NLINES];
#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
        {
            val[i].x = reduce_id;
            val[i].y = reduce_id;
            val[i].z = reduce_id;
            val[i].w = reduce_id;
        }

#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            MULTIMEM_ST(val[i], mc_ptr + (line + i * loop_step0 + myrank * numlines));
    }

    for (int line = start_elem; line < end_elem; line += loop_step)
    {
        uint4 val[UNROLL_NLINES];
        {
            bool readAgain;
            do
            {
                readAgain = false;
#pragma unroll
                for (int i = 0; i < UNROLL_NLINES; i++)
                {
                    load128(buff_ptr + (line + i * loop_step0), val[i]);
                    readAgain |= ((threadIdx.x % 8) == 7) && (val[i].w != reduce_id);
                }
            } while (__any_sync(0xffffffff, readAgain));
        }
    }

} // quant kernel fp16->fp8 oneshot(LL style)

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_res_allgather(int const op, int const flagoffset, int const firstrank,
        int const myrank, int const gpustep, size_t const lineoffset, int const numlines, void** commbuff,
        int const handleridx, float4* mc_ptr, int const RANKS, uint4* residual_in, int res_offset)
{
    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = (*reduceidptr) + 1;
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
    }
    __syncthreads();

    int const loop_step0 = blockDim.x;
    int const loop_step = loop_step0 * UNROLL_NLINES * gridDim.x;
    int const start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL_NLINES;
    int const end_elem = max(start_elem, numlines);

    for (int line = start_elem; line < end_elem; line += loop_step)
    {
        uint4 val[UNROLL_NLINES];

#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            val[i] = residual_in[res_offset + line + i * loop_step0];

#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            MULTIMEM_ST(val[i], mc_ptr + (lineoffset + line + i * loop_step0));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        int volatile* flag = (int volatile*) &myptr[targetgpu];
#ifdef UB_TIMEOUT_ENABLED
        clock_t s = clock64();
#endif
        while (*flag < reduce_id)
        {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > 2ull * TIMEOUT)
            {
                printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
                break;
            }
#endif
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // residual allgather kernel

#else
template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, float2* mc_ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_res_allgather(int const op, int const flagoffset, int const firstrank,
        int const myrank, int const gpustep, size_t const lineoffset, int const numlines, void** commbuff,
        int const handleridx, float4* mc_ptr, int const RANKS, uint4* residual_in, int res_offset)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, uint2* ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot_lamport(
    int const myrank, uint4* ptr_in, int const numlines, int* reduceidptr, uint4* buff_ptr, float4* mc_ptr,
    DType const* beta, DType const* gamma, float const eps, int const RANKS, uint2* ptr_out,
    size_t const out_lineoffset, float const* scale, uint4* residual_in, uint4* residual_out)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}
#endif

#define callranksMC_RMSNORM_QUANT(x)                                                                                   \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = op - MAX_OPS,                                                                                       \
            arg2 = REG0_OFFSET(comm) - (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,   \
            arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step;                                                      \
        size_t arg6 = offset / 8 + first_token * hidden_lines;                                                         \
        int arg7 = hidden_lines * my_tokens;                                                                           \
        void** arg8 = (void**) (comm->gpu_ptrs);                                                                       \
        int arg9 = handler * comm->nvsize;                                                                             \
        void* arg10 = comm->mc_ptr[handler];                                                                           \
        DType* arg11 = (DType*) beta;                                                                                  \
        DType* arg12 = (DType*) gamma;                                                                                 \
        float arg13 = eps;                                                                                             \
        int arg14 = ar_nvsize;                                                                                         \
        void* arg15 = comm->mc_ptr[out_handler];                                                                       \
        size_t arg16 = out_offset / 8 + first_token * hidden_lines;                                                    \
        float* arg17 = scalefactor;                                                                                    \
        void* arg18 = residual_in;                                                                                     \
        void* arg19 = residual_out;                                                                                    \
        int arg20 = first_token * hidden_lines;                                                                        \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14),         \
            reinterpret_cast<void*>(&arg15), reinterpret_cast<void*>(&arg16), reinterpret_cast<void*>(&arg17),         \
            reinterpret_cast<void*>(&arg18), reinterpret_cast<void*>(&arg19), reinterpret_cast<void*>(&arg20)};        \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant<DType, x>), kernelArgs));                 \
    }

#define callranksMC_RMSNORM_QUANT_ONESHOT(x)                                                                           \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = op - MAX_OPS,                                                                                       \
            arg2 = REG0_OFFSET(comm) - (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,   \
            arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step;                                                      \
        size_t arg6 = offset / 8;                                                                                      \
        int arg7 = elements / 8;                                                                                       \
        void** arg8 = (void**) (comm->gpu_ptrs);                                                                       \
        int arg9 = handler * comm->nvsize;                                                                             \
        void* arg10 = comm->mc_ptr[handler];                                                                           \
        DType* arg11 = (DType*) beta;                                                                                  \
        DType* arg12 = (DType*) gamma;                                                                                 \
        float arg13 = eps;                                                                                             \
        int arg14 = ar_nvsize;                                                                                         \
        void* arg15 = comm->mem_ptr[out_handler];                                                                      \
        size_t arg16 = out_offset / 8;                                                                                 \
        float* arg17 = scalefactor;                                                                                    \
        void* arg18 = residual_in;                                                                                     \
        void* arg19 = residual_out;                                                                                    \
        int arg20 = 0;                                                                                                 \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14),         \
            reinterpret_cast<void*>(&arg15), reinterpret_cast<void*>(&arg16), reinterpret_cast<void*>(&arg17),         \
            reinterpret_cast<void*>(&arg18), reinterpret_cast<void*>(&arg19), reinterpret_cast<void*>(&arg20)};        \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot<DType, x>), kernelArgs));         \
    }

#define callranksMC_RMSNORM_QUANT_ONESHOT_LL(x)                                                                        \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = ar_nvrank;                                                                                          \
        void* arg2 = reinterpret_cast<uint8_t*>(comm->mem_ptr[handler]) + (offset * 2);                                \
        int arg3 = elements / 8;                                                                                       \
        void* arg4                                                                                                     \
            = reinterpret_cast<uint8_t*>(comm->mem_ptr[0]) + (REG0_OFFSET(comm) - REG0_SINGLENODE) * sizeof(int);      \
        void* arg5 = reinterpret_cast<uint8_t*>(comm->mem_ptr[0]) + sizeof(int) * (REG0_OFFSET(comm) + REG0_FLAGS);    \
        void* arg6 = reinterpret_cast<uint8_t*>(comm->mc_ptr[0]) + sizeof(int) * (REG0_OFFSET(comm) + REG0_FLAGS);     \
        DType* arg7 = (DType*) beta;                                                                                   \
        DType* arg8 = (DType*) gamma;                                                                                  \
        float arg9 = eps;                                                                                              \
        int arg10 = ar_nvsize;                                                                                         \
        void* arg11 = comm->mem_ptr[out_handler];                                                                      \
        size_t arg12 = out_offset / 8;                                                                                 \
        float* arg13 = scalefactor;                                                                                    \
        void* arg14 = residual_in;                                                                                     \
        void* arg15 = residual_out;                                                                                    \
        void* kernelArgs[]                                                                                             \
            = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2), reinterpret_cast<void*>(&arg3),         \
                reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5), reinterpret_cast<void*>(&arg6),        \
                reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8), reinterpret_cast<void*>(&arg9),        \
                reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11), reinterpret_cast<void*>(&arg12),     \
                reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14), reinterpret_cast<void*>(&arg15)};    \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot_lamport<DType, x>), kernelArgs)); \
    }

#define callranksMC_RES_AG(x)                                                                                          \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = op - MAX_OPS,                                                                                       \
            arg2 = REG0_OFFSET(comm) - (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,   \
            arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step;                                                      \
        size_t arg6 = offset / 8 + first_token * hidden_lines;                                                         \
        int arg7 = hidden_lines * my_tokens;                                                                           \
        void** arg8 = (void**) (comm->gpu_ptrs);                                                                       \
        int arg9 = handler * comm->nvsize;                                                                             \
        void* arg10 = comm->mc_ptr[handler];                                                                           \
        int arg11 = ar_nvsize;                                                                                         \
        uint4* arg12 = (uint4*) residual_in;                                                                           \
        int arg13 = first_token * hidden_lines;                                                                        \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13)};                                         \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_inplace_gpu_mc_res_allgather<DType, x>), kernelArgs));                 \
    }

template <typename DType>
int allreduce2_userbuff_inplace_gpu(int const maxcredit, int const handler, size_t const offset, size_t const elements,
    int const blocksize, communicator* comm, cudaStream_t stream, int op)
{
    // schedule GPU kernel only
    // CPU/SHARP part is responsibility of caller
    int const ar_firstgpu = op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
    int const ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
    int const ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
    int const ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

    if (elements < 8)
        return 0;
    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int warps = comm->threads / 32;
    if (warps < ar_nvsize)
        warps = ar_nvsize;
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    if (op == userbuffers_allreduceop_nonsharp2 && comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        callranksMC(2) callranksMC(4) callranksMC(8)
#ifdef MNNVL
            callranksMC(16) callranksMC(32)
#endif
    }
    else
    {
        callranks(2) callranks(4) callranks(8)
#ifdef MNNVL
            callranks(16) callranks(32)
#endif
    }

    return sms;
}

template <typename DType>
void allreduce_nonsharp_inplace(
    int const handler, size_t const offset, size_t const elements, communicator* comm, cudaStream_t stream, int op)
{
    if (elements < 64)
        return;
    int blocksize = elements * 2;
    int maxcredit = 0;
    int sms = allreduce2_userbuff_inplace_gpu<DType>(maxcredit, handler, offset, elements, blocksize, comm, stream, op);
}

template <typename DType>
void allreduce2_userbuff_inplace(
    int const handler, size_t const offset, size_t const elements, communicator* comm, cudaStream_t stream)
{
    allreduce_nonsharp_inplace<DType>(handler, offset, elements, comm, stream, userbuffers_allreduceop_nonsharp2);
}

template <typename DType>
int allreduce2_userbuff_inplace_rmsnorm_quant(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, communicator* comm, cudaStream_t stream)
{
    // schedule GPU kernel only
    // CPU/SHARP part is not supported yet;
    int op = userbuffers_allreduceop_nonsharp2;
    int const ar_firstgpu = op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
    int const ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
    int const ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
    int const ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

    if (elements % hidden_size)
        return 0;
    assert(hidden_size % 8 == 0);
    int hidden_lines = hidden_size / 8;
    shard_tokens(elements / hidden_size, ar_nvsize, ar_nvrank);

    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int nthreads = hidden_size / 8;
    int nlines = 1;
    while (nthreads > 1024)
    {
        nlines++;
        assert(nlines <= 4);
        if ((hidden_size / 8) % nlines == 0)
            nthreads = ((hidden_size / 8)) / nlines;
    }

    SETUP_LAUNCH_CONFIG(sms, nthreads, stream);
    if (op == userbuffers_allreduceop_nonsharp2 && comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        if (comm->oneshot > 1 || (comm->oneshot == 1 && (elements * ar_nvsize <= 131072)))
        {
            if (comm->oneshot < 3)
            {
                callranksMC_RMSNORM_QUANT_ONESHOT(1) callranksMC_RMSNORM_QUANT_ONESHOT(2)
                    callranksMC_RMSNORM_QUANT_ONESHOT(3) callranksMC_RMSNORM_QUANT_ONESHOT(4)
            }
            else
            {
                sms = 1;
                callranksMC_RMSNORM_QUANT_ONESHOT_LL(1) callranksMC_RMSNORM_QUANT_ONESHOT_LL(2)
            }
        }
        else
        {
            callranksMC_RMSNORM_QUANT(1) callranksMC_RMSNORM_QUANT(2) callranksMC_RMSNORM_QUANT(3)
                callranksMC_RMSNORM_QUANT(4)
        }
    }
    else
    {
        assert(0);
    }

    return sms;
}

template <typename DType>
int allgather2_userbuff_residual(int const handler, size_t const offset, size_t const elements, int const hidden_size,
    void* residual_in, communicator* comm, cudaStream_t stream)
{
    // schedule GPU kernel only
    // CPU/SHARP part is not supported yet;
    if (comm->oneshot == 2 || (comm->oneshot == 1 && (elements * comm->ar2_nvsize <= 131072)))
    {
        TLLM_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(comm->mem_ptr[handler]) + (offset * 2), residual_in,
            elements * 2, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }
    int op = userbuffers_allreduceop_nonsharp2;
    int const ar_firstgpu = op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
    int const ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
    int const ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
    int const ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

    if (elements % hidden_size)
        return 0;
    assert(hidden_size % 8 == 0);
    int hidden_lines = hidden_size / 8;
    shard_tokens(elements / hidden_size, ar_nvsize, ar_nvrank);

    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int nthreads = hidden_size / 8;
    int nlines = 1;
    while (nthreads > 1024)
    {
        nlines++;
        assert(nlines <= 4);
        if ((hidden_size / 8) % nlines == 0)
            nthreads = ((hidden_size / 8)) / nlines;
    }
    SETUP_LAUNCH_CONFIG(sms, nthreads, stream);
    if (op == userbuffers_allreduceop_nonsharp2 && comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        callranksMC_RES_AG(1) callranksMC_RES_AG(2) callranksMC_RES_AG(3) callranksMC_RES_AG(4)
    }
    else
    {
        assert(0);
    }

    return sms;
}

void allreduce2_userbuff_inplace_impl(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF: allreduce2_userbuff_inplace<half>(handler, offset, elements, comm, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        allreduce2_userbuff_inplace<__nv_bfloat16>(handler, offset, elements, comm, stream);
        break;
#endif
    default: TLLM_THROW("Unsupported dataType for allreduce2_userbuff_inplace_impl");
    }
}

int allgather2_userbuff_residual_impl(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        return allgather2_userbuff_residual<half>(handler, offset, elements, hidden_size, residual, comm, stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        return allgather2_userbuff_residual<__nv_bfloat16>(
            handler, offset, elements, hidden_size, residual, comm, stream);
        break;
#endif
    default: TLLM_THROW("Unsupported dataType for allgather2_userbuff_residual_impl");
    }
}

int allreduce2_userbuff_inplace_rmsnorm_quant_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        return allreduce2_userbuff_inplace_rmsnorm_quant<half>(handler, offset, out_handler, out_offset, elements,
            hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, comm, stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        return allreduce2_userbuff_inplace_rmsnorm_quant<__nv_bfloat16>(handler, offset, out_handler, out_offset,
            elements, hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, comm, stream);
        break;
#endif
    default: TLLM_THROW("Unsupported dataType for allreduce2_userbuff_inplace_rmsnorm_quant_impl");
    }
}
} // namespace tensorrt_llm::kernels::ub
