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

#include "tensorrt_llm/kernels/quantization.cuh"
#include "userbuffers.h"
#include "utils.h"

namespace tensorrt_llm::kernels::ub
{
using namespace tensorrt_llm::runtime::ub;
#define MAX_THREADS 1024
#define TIMEOUT 200000000000ull

__forceinline__ __device__ int prev_flag(int flag)
{
    return flag > 0 ? (flag - 1) : 2;
}

__forceinline__ __device__ int next_flag(int flag)
{
    return flag < 2 ? (flag + 1) : 0;
}

__forceinline__ __device__ void multi_gpu_block_barrier(int reduce_id, int volatile* flag)
{
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (*flag == prev_flag(reduce_id))
    {
#ifdef UB_TIMEOUT_ENABLED
        if (clock64() - s > 2ull * TIMEOUT)
        {
            printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id, *flag);
            break;
        }
#endif
    }
}

template <typename DType, int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw(int const op, int const flagoffset, int const firstrank, int const myrank,
        int const gpustep, size_t const lineoffset, int const numlines, void** commbuff, int const handleridx)
{
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
#if __CUDA_ARCH__ >= 900
        cudaGridDependencySynchronize();
#endif
        flagptr[physgpu] = reduce_id;
        userptr[threadIdx.x] = reinterpret_cast<int4*>(commbuff[targetgpu + handleridx]);
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
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
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Hopper)

template <typename DType, int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr(int const op, int const flagoffset, int const firstrank, int const myrank,
        int const gpustep, size_t const lineoffset, int const numlines, void** commbuff, int const handleridx)
{
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
#if __CUDA_ARCH__ >= 900
        cudaGridDependencySynchronize();
#endif
        flagptr[physgpu] = reduce_id;
        userptr[threadIdx.x] = reinterpret_cast<int4*>(commbuff[targetgpu + handleridx]);
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
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
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
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

#if __CUDA_ARCH__ >= 900
template <typename ValType, typename PtrType>
__device__ __forceinline__ void MULTIMEM_ST(ValType val, PtrType ptr)
{
    asm volatile(
        "multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
}

template <>
__device__ __forceinline__ void MULTIMEM_ST<uint32_t, uint32_t*>(uint32_t val, uint32_t* ptr)
{
    asm volatile("multimem.st.global.b32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

template <typename ValType, typename PtrType>
__device__ __forceinline__ void MULTIMEM_ST2(ValType& val, PtrType ptr)
{
    asm volatile("multimem.st.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y) : "memory");
}

template <typename DType, bool const DISABLE_FP32_ACC, typename ValType, typename PtrType>
__device__ __forceinline__ void MULTIMEM_LD(ValType& val, PtrType ptr)
{
    if constexpr (std::is_same_v<DType, half>)
    {
        if (!DISABLE_FP32_ACC)
        {
            asm("multimem.ld_reduce.global.add.v4.f16x2.acc::f32 {%0,%1,%2,%3}, [%4];"
                : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                : "l"(ptr)
                : "memory");
        }
        else
        {
            asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
                : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                : "l"(ptr)
                : "memory");
        }
    }
#ifdef ENABLE_BF16
    if constexpr (std::is_same_v<DType, __nv_bfloat16>)
    {
        if (!DISABLE_FP32_ACC)
        {
            asm("multimem.ld_reduce.global.add.v4.bf16x2.acc::f32 {%0,%1,%2,%3}, [%4];"
                : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                : "l"(ptr)
                : "memory");
        }
        else
        {
            asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                : "l"(ptr)
                : "memory");
        }
    }
#endif
}

// All MC kernels here
template <typename DType, int RANKS, bool DISABLE_FP32_ACC>
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;

        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
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
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));
#pragma unroll
        for (int i = 0; i < UNROLL_MC; i++)
            MULTIMEM_ST(val[i], mc_ptr + (lineoffset + line + i * loop_step0));
    }
    for (int line = end_aligned; line < end_elem; line += loop_step0)
    {
        uint4 val;
        MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val, mc_ptr + (lineoffset + line));
        MULTIMEM_ST(val, mc_ptr + (lineoffset + line));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Hopper) MC

#else
template <typename DType, int RANKS, bool DISABLE_FP32_ACC>
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
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_inplace_gpu_mc<DType, x, DISABLE_FP32_ACC>), kernelArgs));             \
    }

struct LaunchConfig
{
    LaunchConfig(communicator* comm, int sms, int threads, cudaStream_t stream)
    {
        cfg.gridDim = sms;
        cfg.blockDim = threads;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        attribute[0].id = cudaLaunchAttributeCooperative;
        attribute[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[1].val.programmaticStreamSerializationAllowed = comm->pdl_launch;
        attribute[2].id = cudaLaunchAttributeClusterDimension;
        attribute[2].val.clusterDim.x = sms % comm->cga_size == 0 ? comm->cga_size : 1;
        attribute[2].val.clusterDim.y = 1;
        attribute[2].val.clusterDim.z = 1;
        cfg.attrs = attribute;
        cfg.numAttrs = comm->sm_arch >= 9 ? 3 : 1;
    }

    cudaLaunchConfig_t& get()
    {
        return cfg;
    }

    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[3];
};

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

#define SHARD_TOKENS(ntokens, nranks, myrank)                                                                          \
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

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4_mc(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Get absolute maximum values among the local 8 values.
    auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
    for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++)
    {
        localMax = __hmax2(localMax, __habs2(vec.elts[i]));
    }

    // Get the absolute maximum among all 16 values (two threads).
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    // Get the final absolute maximum values.
    float vecMax = float(__hmax(localMax.x, localMax.y));

    // Get the SF (max value of the vector / max value of e2m1).
    // maximum value of e2m1 = 6.0.
    // TODO: use half as compute data type.
    float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    // 8 bits representation of the SF.
    uint8_t fp8SFVal;
    // Write the SF to global memory (STG.8).
    if constexpr (UE8M0_SF)
    {
        // Extract the 8 exponent bits from float32.
        // float 32bits = 1 sign bit + 8 exponent bits + 23 mantissa bits.
        uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
        fp8SFVal = tmp & 0xff;
        // Convert back to fp32.
        reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
    }
    else
    {
        // Here SFValue is always positive, so E4M3 is the same as UE4M3.
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
        // Convert back to fp32.
        SFValue = float(tmp);
    }
    // Get the output scale.
    // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
    float outputScale
        = SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

    if (threadIdx.x % 2 == 0)
    {
        // Write the SF to global memory (STG.8).
        // *SFout = fp8SFVal;
        uint32_t SFValVec4 = 0;
        uint8_t* SFPtr = reinterpret_cast<uint8_t*>(&SFValVec4);
        SFPtr[(threadIdx.x % 8) / 2] = fp8SFVal;
        SFValVec4 |= __shfl_xor_sync(0x55555555, SFValVec4, 2);
        SFValVec4 |= __shfl_xor_sync(0x55555555, SFValVec4, 4);
        if (threadIdx.x % 8 == 0)
        {
            MULTIMEM_ST(SFValVec4, reinterpret_cast<uint32_t*>(SFout));
        }
    }

    // Convert the input to float.
    float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
    for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++)
    {
        if constexpr (std::is_same_v<Type, half>)
        {
            fp2Vals[i] = __half22float2(vec.elts[i]);
        }
        else
        {
            fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
        }
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    // Convert to e2m1 values.
    uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

    // Write the e2m1 values to global memory.
    return e2m1Vec;
#else
    return 0;
#endif
}

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_fp4(int const op, int const flagoffset, int const firstrank,
        int const myrank, int const gpustep, size_t const lineoffset, int const numlines, void** commbuff,
        int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma, float const eps, int const RANKS,
        uint32_t* mc_ptr_out, size_t const out_lineoffset, float const* scale, uint4* residual_in, uint4* residual_out,
        int res_offset, uint32_t* scale_out, size_t const scale_out_offset, int first_token)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int SF_VEC_SIZE = 16;
    using PackedVec = PackedVec<DType>;
    cudaTriggerProgrammaticLaunchCompletion();
    float sf = *scale;
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
    }
    __syncthreads();

    int const loop_step0 = blockDim.x;
    int const loop_step = loop_step0 * UNROLL_NLINES * gridDim.x;
    int const start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL_NLINES;
    int const end_elem = max(start_elem, numlines);
    int token_idx = first_token + blockIdx.x;
    for (int line = start_elem; line < end_elem; line += loop_step, token_idx += gridDim.x)
    {
        uint4 val[UNROLL_NLINES];
        DType* x = reinterpret_cast<DType*>(&val[0]);
#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

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
        PackedVec valout;
        DType* y = reinterpret_cast<DType*>(&valout);
#pragma unroll
        for (int g = 0; g < UNROLL_NLINES; g++)
        {
#pragma unroll
            for (int j = 0; j < sizeof(int4) / sizeof(DType); j++)
            {
                y[j] = static_cast<DType>(compute_rmsnorm2<DType>((float) x[i], s_variance, gamma, beta,
                    (threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(DType) + j));
                i++;
            }
            uint8_t* sf_out = nullptr;
            if (threadIdx.x % 8 == 0)
            {
                sf_out = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt /* batchIdx */, token_idx,
                    threadIdx.x + g * loop_step0, std::nullopt /* numRows */, hidden_dim / SF_VEC_SIZE,
                    scale_out + scale_out_offset, QuantizationSFLayout::SWIZZLED);
            }
            uint32_t val = cvt_warp_fp16_to_fp4_mc<DType, SF_VEC_SIZE>(valout, sf, sf_out);
            MULTIMEM_ST(val, mc_ptr_out + (out_lineoffset + line + g * loop_step0));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
#endif
}

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_fp4_oneshot(int const op, int const flagoffset,
        int const firstrank, int const myrank, int const gpustep, size_t const lineoffset, int const numlines,
        void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma, float const eps,
        int const RANKS, uint32_t* mc_ptr_out, size_t const out_lineoffset, float const* scale, uint4* residual_in,
        uint4* residual_out, int res_offset, uint32_t* scale_out, size_t const scale_out_offset)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int SF_VEC_SIZE = 16;
    using PackedVec = PackedVec<DType>;
    cudaTriggerProgrammaticLaunchCompletion();
    float sf = *scale;
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    __syncthreads();

    int const loop_step0 = blockDim.x;
    int const loop_step = loop_step0 * UNROLL_NLINES * gridDim.x;
    int const start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL_NLINES;
    int const end_elem = max(start_elem, numlines);
    int token_idx = blockIdx.x;
    for (int line = start_elem; line < end_elem; line += loop_step, token_idx += gridDim.x)
    {
        uint4 val[UNROLL_NLINES];
        DType* x = reinterpret_cast<DType*>(&val[0]);
#pragma unroll
        for (int i = 0; i < UNROLL_NLINES; i++)
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

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
        PackedVec valout;
        DType* y = reinterpret_cast<DType*>(&valout);

#pragma unroll
        for (int g = 0; g < UNROLL_NLINES; g++)
        {
#pragma unroll
            for (int j = 0; j < sizeof(int4) / sizeof(DType); j++)
            {
                y[j] = static_cast<DType>(compute_rmsnorm2<DType>((float) x[i], s_variance, gamma, beta,
                    (threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(DType) + j));
                i++;
            }
            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt /* batchIdx */, token_idx,
                threadIdx.x + g * loop_step0, std::nullopt /* numRows */, hidden_dim / SF_VEC_SIZE,
                scale_out + scale_out_offset, QuantizationSFLayout::SWIZZLED);
            mc_ptr_out[out_lineoffset + line + g * loop_step0]
                = cvt_warp_fp16_to_fp4<DType, SF_VEC_SIZE, false>(valout, sf, sf_out);
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
#endif
}

#if __CUDA_ARCH__ >= 900

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_gpu_mc_rmsnorm(int const op, int const flagoffset, int const firstrank, int const myrank,
        int const gpustep, const size_t lineoffset, int const numlines, void** commbuff, int const handleridx,
        float4* mc_ptr, DType const* beta, DType const* gamma, float const eps, int const RANKS, float4* mc_ptr_out,
        size_t const out_lineoffset, uint4* residual_in, uint4* residual_out, int res_offset)
{
    cudaTriggerProgrammaticLaunchCompletion();
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
        reduceidptr = myptr - MAX_OPS; //+op;
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
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
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

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
#pragma unroll
        for (int g = 0; g < UNROLL_NLINES; g++)
        {
#pragma unroll
            for (int j = 0; j < sizeof(int4) / sizeof(DType); j++)
            {
                x[i] = cuda_cast<DType>(compute_rmsnorm2<DType>((float) (x[i]), s_variance, gamma, beta,
                    (threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(DType) + j));
                i++;
            }
            MULTIMEM_ST(val[g], mc_ptr_out + (out_lineoffset + line + g * loop_step0));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
    __syncthreads();

    if (threadIdx.x < RANKS)
    {
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Hopper) MC with rmsNorm fused

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_gpu_mc_rmsnorm_oneshot(int const op, int const flagoffset, int const firstrank,
        int const myrank, int const gpustep, const size_t lineoffset, int const numlines, void** commbuff,
        int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma, float const eps, int const RANKS,
        uint4* uc_ptr_out, size_t const out_lineoffset, uint4* residual_in, uint4* residual_out, int res_offset)
{
    cudaTriggerProgrammaticLaunchCompletion();
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
        reduceidptr = myptr - MAX_OPS; //+op;
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
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
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

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
#pragma unroll
        for (int g = 0; g < UNROLL_NLINES; g++)
        {
#pragma unroll
            for (int j = 0; j < sizeof(int4) / sizeof(DType); j++)
            {
                x[i] = cuda_cast<DType>(compute_rmsnorm2<DType>((float) (x[i]), s_variance, gamma, beta,
                    (threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(DType) + j));
                i++;
            }
            uc_ptr_out[out_lineoffset + line + g * loop_step0] = val[g];
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // fp16 inplace reduce kernel (Hopper) MC with rmsNorm fused oneshot

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, float2* mc_ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    cudaTriggerProgrammaticLaunchCompletion();
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
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
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

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
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // quant kernel fp16->fp8 twoshot

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, uint2* mc_ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    cudaTriggerProgrammaticLaunchCompletion();
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
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
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
            MULTIMEM_LD<DType, DISABLE_FP32_ACC>(val[i], mc_ptr + (lineoffset + line + i * loop_step0));

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

template <typename DType, int UNROLL_NLINES>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_res_allgather(int const op, int const flagoffset, int const firstrank,
        int const myrank, int const gpustep, size_t const lineoffset, int const numlines, void** commbuff,
        int const handleridx, float4* mc_ptr, int const RANKS, uint4* residual_in, int res_offset)
{
    cudaTriggerProgrammaticLaunchCompletion();
    int *flagptr, physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    if (threadIdx.x < RANKS)
    {
        physgpu = myrank * gpustep + firstrank;
        targetgpu = threadIdx.x * gpustep + firstrank;
        int const blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
        myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
        reduceidptr = myptr - MAX_OPS;
        reduce_id = next_flag(*reduceidptr);
        flagptr = (reinterpret_cast<int*>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
        myptr += blockflagoffset;
        cudaGridDependencySynchronize();
        flagptr[physgpu] = reduce_id;
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
        reduce_id = next_flag(reduce_id);
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
        multi_gpu_block_barrier(reduce_id, (int volatile*) &myptr[targetgpu]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *reduceidptr = reduce_id;
} // residual allgather kernel

#else
template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_gpu_mc_rmsnorm(int const op, int const flagoffset, int const firstrank, int const myrank,
        int const gpustep, const size_t lineoffset, int const numlines, void** commbuff, int const handleridx,
        float4* mc_ptr, DType const* beta, DType const* gamma, float const eps, int const RANKS, float4* uc_ptr_out,
        size_t const out_lineoffset, uint4* residual_in, uint4* residual_out, int res_offset)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_gpu_mc_rmsnorm_oneshot(int const op, int const flagoffset, int const firstrank,
        int const myrank, int const gpustep, const size_t lineoffset, int const numlines, void** commbuff,
        int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma, float const eps, int const RANKS,
        uint4* uc_ptr_out, size_t const out_lineoffset, uint4* residual_in, uint4* residual_out, int res_offset)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
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

template <typename DType, int UNROLL_NLINES, bool DISABLE_FP32_ACC>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot(int const op,
    int const flagoffset, int const firstrank, int const myrank, int const gpustep, size_t const lineoffset,
    int const numlines, void** commbuff, int const handleridx, float4* mc_ptr, DType const* beta, DType const* gamma,
    float const eps, int const RANKS, uint2* mc_ptr_out, size_t const out_lineoffset, float const* scale,
    uint4* residual_in, uint4* residual_out, int res_offset)
{
    printf("userbuffer based kernels not implemented when SM < 90\n");
    asm volatile("brkpt;\n");
}

#endif

#define callranksMC_RMSNORM_QUANT(x)                                                                                   \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(&cfg,                                                                      \
            (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant<DType, x, DISABLE_FP32_ACC>), kernelArgs));     \
    }

#define callranksMC_RMSNORM_QUANT_ONESHOT(x)                                                                           \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(&cfg,                                                                      \
            (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_oneshot<DType, x, DISABLE_FP32_ACC>),           \
            kernelArgs));                                                                                              \
    }

#define callranksMC_RMSNORM_QUANT_FP4(x)                                                                               \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        size_t arg16 = out_offset / 4 + first_token * hidden_lines;                                                    \
        float* arg17 = scalefactor;                                                                                    \
        void* arg18 = residual_in;                                                                                     \
        void* arg19 = residual_out;                                                                                    \
        int arg20 = first_token * hidden_lines;                                                                        \
        void* arg21 = comm->mc_ptr[scale_handler];                                                                     \
        size_t arg22 = scale_offset / 4;                                                                               \
        int arg23 = first_token;                                                                                       \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14),         \
            reinterpret_cast<void*>(&arg15), reinterpret_cast<void*>(&arg16), reinterpret_cast<void*>(&arg17),         \
            reinterpret_cast<void*>(&arg18), reinterpret_cast<void*>(&arg19), reinterpret_cast<void*>(&arg20),         \
            reinterpret_cast<void*>(&arg21), reinterpret_cast<void*>(&arg22), reinterpret_cast<void*>(&arg23)};        \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(&cfg,                                                                      \
            (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_fp4<DType, x, DISABLE_FP32_ACC>), kernelArgs)); \
    }

#define callranksMC_RMSNORM_QUANT_FP4_ONESHOT(x)                                                                       \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        size_t arg16 = out_offset / 4;                                                                                 \
        float* arg17 = scalefactor;                                                                                    \
        void* arg18 = residual_in;                                                                                     \
        void* arg19 = residual_out;                                                                                    \
        int arg20 = 0;                                                                                                 \
        void* arg21 = reinterpret_cast<uint8_t*>(comm->ucbase_ptr[scale_handler])                                      \
            + (ar_firstgpu + ar_nvrank) * comm->mem_size[scale_handler];                                               \
        size_t arg22 = scale_offset / 4;                                                                               \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14),         \
            reinterpret_cast<void*>(&arg15), reinterpret_cast<void*>(&arg16), reinterpret_cast<void*>(&arg17),         \
            reinterpret_cast<void*>(&arg18), reinterpret_cast<void*>(&arg19), reinterpret_cast<void*>(&arg20),         \
            reinterpret_cast<void*>(&arg21), reinterpret_cast<void*>(&arg22)};                                         \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(&cfg,                                                                      \
            (void*) (userbuffers_fp16_sum_inplace_gpu_mc_rmsnorm_quant_fp4_oneshot<DType, x, DISABLE_FP32_ACC>),       \
            kernelArgs));                                                                                              \
    }
#define callranksMC_RES_AG(x)                                                                                          \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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

#define callranksMC_RMSNORM(x)                                                                                         \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        void* arg17 = residual_in;                                                                                     \
        void* arg18 = residual_out;                                                                                    \
        int arg19 = first_token * hidden_lines;                                                                        \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14),         \
            reinterpret_cast<void*>(&arg15), reinterpret_cast<void*>(&arg16), reinterpret_cast<void*>(&arg17),         \
            reinterpret_cast<void*>(&arg18), reinterpret_cast<void*>(&arg19)};                                         \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_gpu_mc_rmsnorm<DType, x, DISABLE_FP32_ACC>), kernelArgs));             \
    }

#define callranksMC_RMSNORM_ONESHOT(x)                                                                                 \
    if (nlines == x)                                                                                                   \
    {                                                                                                                  \
        int arg1 = userbuffers_allreduceop_nonsharp2 - MAX_OPS, arg2 = REG0_OFFSET(comm) - REG0_SINGLENODE + MAX_OPS,  \
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
        void* arg17 = residual_in;                                                                                     \
        void* arg18 = residual_out;                                                                                    \
        int arg19 = 0;                                                                                                 \
        void* kernelArgs[] = {reinterpret_cast<void*>(&arg1), reinterpret_cast<void*>(&arg2),                          \
            reinterpret_cast<void*>(&arg3), reinterpret_cast<void*>(&arg4), reinterpret_cast<void*>(&arg5),            \
            reinterpret_cast<void*>(&arg6), reinterpret_cast<void*>(&arg7), reinterpret_cast<void*>(&arg8),            \
            reinterpret_cast<void*>(&arg9), reinterpret_cast<void*>(&arg10), reinterpret_cast<void*>(&arg11),          \
            reinterpret_cast<void*>(&arg12), reinterpret_cast<void*>(&arg13), reinterpret_cast<void*>(&arg14),         \
            reinterpret_cast<void*>(&arg15), reinterpret_cast<void*>(&arg16), reinterpret_cast<void*>(&arg17),         \
            reinterpret_cast<void*>(&arg18), reinterpret_cast<void*>(&arg19)};                                         \
        TLLM_CUDA_CHECK(cudaLaunchKernelExC(                                                                           \
            &cfg, (void*) (userbuffers_fp16_sum_gpu_mc_rmsnorm_oneshot<DType, x, DISABLE_FP32_ACC>), kernelArgs));     \
    }

template <typename DType, bool DISABLE_FP32_ACC>
int allreduce2_userbuff_inplace_gpu(int const maxcredit, int const handler, size_t const offset, size_t const elements,
    int const blocksize, communicator* comm, cudaStream_t stream)
{
    // schedule GPU kernel only
    // CPU/SHARP part is responsibility of caller
    int const ar_firstgpu = comm->tp_first_rank;
    int const ar_step = 1;
    int const ar_nvsize = comm->tp_size;
    int const ar_nvrank = comm->tp_rank;

    if (elements < 8)
        return 0;
    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int warps = comm->threads / 32;
    if (warps < ar_nvsize)
        warps = ar_nvsize;
    LaunchConfig launch_config(comm, sms, warps * 32, stream);
    auto& cfg = launch_config.get();
    if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
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

template <typename DType, bool DISABLE_FP32_ACC>
void allreduce_nonsharp_inplace(
    int const handler, size_t const offset, size_t const elements, communicator* comm, cudaStream_t stream)
{
    if (elements < 64)
        return;
    int blocksize = elements * 2;
    int maxcredit = 0;
    int sms;
    if (DISABLE_FP32_ACC)
    {
        sms = allreduce2_userbuff_inplace_gpu<DType, true>(
            maxcredit, handler, offset, elements, blocksize, comm, stream);
    }
    else
    {
        sms = allreduce2_userbuff_inplace_gpu<DType, false>(
            maxcredit, handler, offset, elements, blocksize, comm, stream);
    }
}

template <typename DType, bool DISABLE_FP32_ACC>
void allreduce2_userbuff_inplace(
    int const handler, size_t const offset, size_t const elements, communicator* comm, cudaStream_t stream)
{
    allreduce_nonsharp_inplace<DType, DISABLE_FP32_ACC>(handler, offset, elements, comm, stream);
}

bool use_oneshot_kernel(communicator* comm, size_t elements, int hidden_size)
{
    TLLM_CHECK(elements % hidden_size == 0);
    int token_num = elements / hidden_size;
    if (comm->oneshot == 1 && (elements * comm->tp_size <= 131072))
    {
        return true;
    }
    else if (comm->oneshot == 2 && token_num <= comm->oneshot_force_enable_threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <typename DType, bool DISABLE_FP32_ACC>
int allreduce2_userbuff_rmsnorm(int const handler, int const offset, int const out_handler, size_t const out_offset,
    int const elements, int const hidden_size, void* beta, void* gamma, float eps, void* residual_in,
    void* residual_out, communicator* comm, cudaStream_t stream)
{
    int const ar_firstgpu = comm->tp_first_rank;
    int const ar_step = 1;
    int const ar_nvsize = comm->tp_size;
    int const ar_nvrank = comm->tp_rank;

    if (elements % hidden_size)
        return 0;
    TLLM_CHECK(hidden_size % 8 == 0);
    int hidden_lines = hidden_size / 8;
    SHARD_TOKENS(elements / hidden_size, ar_nvsize, ar_nvrank);

    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int nthreads = hidden_size / 8;
    int nlines = 1;
    while (nthreads > 1024)
    {
        nlines++;
        TLLM_CHECK(nlines <= 4);
        if ((hidden_size / 8) % nlines == 0)
            nthreads = ((hidden_size / 8)) / nlines;
    }

    LaunchConfig launch_config(comm, sms, nthreads, stream);
    auto& cfg = launch_config.get();
    if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        if (use_oneshot_kernel(comm, elements, hidden_size))
        {
            callranksMC_RMSNORM_ONESHOT(1) callranksMC_RMSNORM_ONESHOT(2) callranksMC_RMSNORM_ONESHOT(3)
                callranksMC_RMSNORM_ONESHOT(4)
        }
        else
        {
            callranksMC_RMSNORM(1) callranksMC_RMSNORM(2) callranksMC_RMSNORM(3) callranksMC_RMSNORM(4)
        }
    }
    else
    {
        TLLM_CHECK(0);
    }

    return sms;
}

template <typename DType, bool DISABLE_FP32_ACC>
int allreduce2_userbuff_inplace_rmsnorm_quant(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, communicator* comm, cudaStream_t stream)
{
    int const ar_firstgpu = comm->tp_first_rank;
    int const ar_step = 1;
    int const ar_nvsize = comm->tp_size;
    int const ar_nvrank = comm->tp_rank;

    if (elements % hidden_size)
        return 0;
    TLLM_CHECK(hidden_size % 8 == 0);
    int hidden_lines = hidden_size / 8;
    SHARD_TOKENS(elements / hidden_size, ar_nvsize, ar_nvrank);

    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int nthreads = hidden_size / 8;
    int nlines = 1;
    while (nthreads > 1024)
    {
        nlines++;
        TLLM_CHECK(nlines <= 4);
        if ((hidden_size / 8) % nlines == 0)
            nthreads = ((hidden_size / 8)) / nlines;
    }

    LaunchConfig launch_config(comm, sms, nthreads, stream);
    auto& cfg = launch_config.get();
    if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        if (use_oneshot_kernel(comm, elements, hidden_size))
        {
            callranksMC_RMSNORM_QUANT_ONESHOT(1) callranksMC_RMSNORM_QUANT_ONESHOT(2)
                callranksMC_RMSNORM_QUANT_ONESHOT(3) callranksMC_RMSNORM_QUANT_ONESHOT(4)
        }
        else
        {
            callranksMC_RMSNORM_QUANT(1) callranksMC_RMSNORM_QUANT(2) callranksMC_RMSNORM_QUANT(3)
                callranksMC_RMSNORM_QUANT(4)
        }
    }
    else
    {
        TLLM_CHECK(0);
    }

    return sms;
}

template <typename DType, bool DISABLE_FP32_ACC>
int allreduce2_userbuff_inplace_rmsnorm_quant_fp4(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, int const scale_handler, size_t const scale_offset, size_t const elements,
    int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor, void* residual_in,
    void* residual_out, communicator* comm, cudaStream_t stream)
{
    int const ar_firstgpu = comm->tp_first_rank;
    int const ar_step = 1;
    int const ar_nvsize = comm->tp_size;
    int const ar_nvrank = comm->tp_rank;

    if (elements % hidden_size)
        return 0;
    TLLM_CHECK(hidden_size % 8 == 0);
    int hidden_lines = hidden_size / 8;
    SHARD_TOKENS(elements / hidden_size, ar_nvsize, ar_nvrank);

    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int nthreads = hidden_size / 8;
    int nlines = 1;
    while (nthreads > 1024)
    {
        nlines++;
        TLLM_CHECK(nlines <= 4);
        if ((hidden_size / 8) % nlines == 0)
            nthreads = ((hidden_size / 8)) / nlines;
    }

    LaunchConfig launch_config(comm, sms, nthreads, stream);
    auto& cfg = launch_config.get();
    if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        if (use_oneshot_kernel(comm, elements, hidden_size))
        {
            callranksMC_RMSNORM_QUANT_FP4_ONESHOT(1) callranksMC_RMSNORM_QUANT_FP4_ONESHOT(2)
                callranksMC_RMSNORM_QUANT_FP4_ONESHOT(3) callranksMC_RMSNORM_QUANT_FP4_ONESHOT(4)
        }
        else
        {
            callranksMC_RMSNORM_QUANT_FP4(1) callranksMC_RMSNORM_QUANT_FP4(2) callranksMC_RMSNORM_QUANT_FP4(3)
                callranksMC_RMSNORM_QUANT_FP4(4)
        }
    }
    else
    {
        TLLM_CHECK(0);
    }

    return sms;
}

template <typename DType>
int allgather2_userbuff_residual(int const handler, size_t const offset, size_t const elements, int const hidden_size,
    void* residual_in, communicator* comm, cudaStream_t stream, bool force_enable)
{
    // schedule GPU kernel only
    // CPU/SHARP part is not supported yet;
    if (!force_enable && use_oneshot_kernel(comm, elements, hidden_size))
    {
        TLLM_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(comm->mem_ptr[handler]) + (offset * 2), residual_in,
            elements * 2, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }
    int const ar_firstgpu = comm->tp_first_rank;
    int const ar_step = 1;
    int const ar_nvsize = comm->tp_size;
    int const ar_nvrank = comm->tp_rank;

    if (elements % hidden_size)
        return 0;
    TLLM_CHECK(hidden_size % 8 == 0);
    int hidden_lines = hidden_size / 8;
    SHARD_TOKENS(elements / hidden_size, ar_nvsize, ar_nvrank);

    int sms = ar_nvsize == 1 ? 2 : comm->sms;
    int nthreads = hidden_size / 8;
    int nlines = 1;
    while (nthreads > 1024)
    {
        nlines++;
        TLLM_CHECK(nlines <= 4);
        if ((hidden_size / 8) % nlines == 0)
            nthreads = ((hidden_size / 8)) / nlines;
    }
    LaunchConfig launch_config(comm, sms, nthreads, stream);
    auto& cfg = launch_config.get();
    if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED))
    {
        callranksMC_RES_AG(1) callranksMC_RES_AG(2) callranksMC_RES_AG(3) callranksMC_RES_AG(4)
    }
    else
    {
        TLLM_CHECK(0);
    }

    return sms;
}

void allreduce2_userbuff_inplace_impl(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            allreduce2_userbuff_inplace<half, true>(handler, offset, elements, comm, stream);
        }
        else
        {
            allreduce2_userbuff_inplace<half, false>(handler, offset, elements, comm, stream);
        }
        break;
    }
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            allreduce2_userbuff_inplace<__nv_bfloat16, true>(handler, offset, elements, comm, stream);
        }
        else
        {
            allreduce2_userbuff_inplace<__nv_bfloat16, false>(handler, offset, elements, comm, stream);
        }
        break;
    }
#endif
    default: TLLM_THROW("Unsupported dataType for allreduce2_userbuff_inplace_impl");
    }
}

int allgather2_userbuff_residual_impl(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream,
    bool force_enable)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        return allgather2_userbuff_residual<half>(
            handler, offset, elements, hidden_size, residual, comm, stream, force_enable);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        return allgather2_userbuff_residual<__nv_bfloat16>(
            handler, offset, elements, hidden_size, residual, comm, stream, force_enable);
        break;
#endif
    default: TLLM_THROW("Unsupported dataType for allgather2_userbuff_residual_impl");
    }
}

int allreduce2_userbuff_rmsnorm_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            return allreduce2_userbuff_rmsnorm<half, true>(handler, offset, out_handler, out_offset, elements,
                hidden_size, beta, gamma, eps, residual_in, residual_out, comm, stream);
        }
        else
        {
            return allreduce2_userbuff_rmsnorm<half, false>(handler, offset, out_handler, out_offset, elements,
                hidden_size, beta, gamma, eps, residual_in, residual_out, comm, stream);
        }
        break;
    }
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            return allreduce2_userbuff_rmsnorm<__nv_bfloat16, true>(handler, offset, out_handler, out_offset, elements,
                hidden_size, beta, gamma, eps, residual_in, residual_out, comm, stream);
        }
        else
        {
            return allreduce2_userbuff_rmsnorm<__nv_bfloat16, false>(handler, offset, out_handler, out_offset, elements,
                hidden_size, beta, gamma, eps, residual_in, residual_out, comm, stream);
        }
        break;
    }
#endif
    default: TLLM_THROW("Unsupported dataType for allreduce2_userbuff_rmsnorm_impl");
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
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant<half, true>(handler, offset, out_handler, out_offset,
                elements, hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, comm, stream);
        }
        else
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant<half, false>(handler, offset, out_handler, out_offset,
                elements, hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, comm, stream);
        }
        break;
    }
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant<__nv_bfloat16, true>(handler, offset, out_handler,
                out_offset, elements, hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, comm,
                stream);
        }
        else
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant<__nv_bfloat16, false>(handler, offset, out_handler,
                out_offset, elements, hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, comm,
                stream);
        }
        break;
    }
#endif
    default: TLLM_THROW("Unsupported dataType for allreduce2_userbuff_inplace_rmsnorm_quant_impl");
    }
}

int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, int const scale_handler, size_t const scale_offset, size_t const elements,
    int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor, void* residual_in,
    void* residual_out, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant_fp4<half, true>(handler, offset, out_handler, out_offset,
                scale_handler, scale_offset, elements, hidden_size, beta, gamma, eps, scalefactor, residual_in,
                residual_out, comm, stream);
        }
        else
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant_fp4<half, false>(handler, offset, out_handler, out_offset,
                scale_handler, scale_offset, elements, hidden_size, beta, gamma, eps, scalefactor, residual_in,
                residual_out, comm, stream);
        }
        break;
    }
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
    {
        if (kDISABLE_FP32_ACCUMULATION)
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant_fp4<__nv_bfloat16, true>(handler, offset, out_handler,
                out_offset, scale_handler, scale_offset, elements, hidden_size, beta, gamma, eps, scalefactor,
                residual_in, residual_out, comm, stream);
        }
        else
        {
            return allreduce2_userbuff_inplace_rmsnorm_quant_fp4<__nv_bfloat16, false>(handler, offset, out_handler,
                out_offset, scale_handler, scale_offset, elements, hidden_size, beta, gamma, eps, scalefactor,
                residual_in, residual_out, comm, stream);
        }
        break;
    }
#endif
    default: TLLM_THROW("Unsupported dataType for allreduce2_userbuff_inplace_rmsnorm_quant_impl");
    }
}
} // namespace tensorrt_llm::kernels::ub
