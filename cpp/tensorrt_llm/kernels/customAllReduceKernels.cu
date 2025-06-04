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

#include "customAllReduceKernels.h"
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include <cooperative_groups.h>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace tensorrt_llm::kernels
{

using tensorrt_llm::common::divUp;
using tensorrt_llm::common::roundUp;

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t ld_flag_acquire(uint32_t* flag_addr)
{
    uint32_t flag;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.global.volatile.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Type Converter that packs data format to 128 bits data type
//
using PackedFloat = union
{
    int4 packed;
    float unpacked[4];
};

using PackedHalf = union
{
    int4 packed;
    half2 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes
{
};

template <>
struct PackedOn16Bytes<float>
{
    using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half>
{
    using Type = PackedHalf;
};

#ifdef ENABLE_BF16
using PackedBFloat16 = union
{
    int4 packed;
    __nv_bfloat162 unpacked[4];
};

template <>
struct PackedOn16Bytes<__nv_bfloat16>
{
    using Type = PackedBFloat16;
};

#endif

__inline__ __device__ bool thread0()
{
    return !threadIdx.x && !threadIdx.y && !threadIdx.z;
}

__inline__ __device__ bool block0()
{
    return !blockIdx.x && !blockIdx.y && !blockIdx.z;
}

// add two 128b data
template <typename T>
inline __device__ int4 add128b(T& a, T& b)
{
    T c;
    c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
    c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
    c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
    c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
    return c.packed;
}

__inline__ __device__ void multi_gpu_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
    size_t const world_size, int const tidx, int const bidx)
{
    // After this function, at least one block in each GPU has reached the barrier
    if (tidx < world_size)
    {
        // we can think of signals having the shape [world_size, world_size]
        // Dimension 0 is the "listening" dimension, dimension 1 is "emitting" dimension

        // Block 0 broadcasts its flag (local_rank on emitting dimension) to all receivers
        size_t offset = (flag % 2) ? world_size : 0;

        if (bidx == 0)
        {
            st_flag_release(flag, signals[tidx] + offset + local_rank);
        }

        // All blocks check that corresponding block 0 on other GPUs have set the flag
        // No deadlock because block #0 is always the first block started
        uint32_t* peer_barrier_d = signals[local_rank] + offset + tidx;
        while (ld_flag_acquire(peer_barrier_d) != flag)
        {
        }
    }

    __syncthreads();
}

__inline__ __device__ void block_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
    size_t const world_size, int const tidx, int const bidx)
{
    // After this function, the block of id == bidx of each GPU has reached the barrier
    if (tidx < world_size)
    {
        // we can think of signals having the shape [world_size, 2, num_blocks, world_size]
        // (+ an offset on dim 2 to account for flags used in multi_gpu_barrier)
        // Dimension 0 is the "listening" dimension, dimension 3 is "emitting" dimension

        // Block broadcast its flag (local_rank on emitting dimension) to all receivers
        uint32_t flag_block_offset = (bidx + 1) * world_size;

        if (flag % 2 == 1)
        {
            flag_block_offset += (MAX_ALL_REDUCE_BLOCKS + 1) * world_size;
        }

        st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);

        // Blocks check that corresponding blocks on other GPUs have also set the flag
        uint32_t* peer_barrier_d = signals[local_rank] + flag_block_offset + tidx;

        while (ld_flag_acquire(peer_barrier_d) != flag)
        {
        }
    }

    __syncthreads();
}

__inline__ __device__ void update_barrier_flag(uint32_t* barrier_flag_ptr, uint32_t* barrier_flag_counter_ptr)
{
    if (thread0())
    {
        atomicAdd(barrier_flag_counter_ptr, 1);

        if (block0())
        {
            auto blockNum = gridDim.x * gridDim.y * gridDim.z;
            while (*reinterpret_cast<uint32_t volatile*>(barrier_flag_counter_ptr) != blockNum)
            {
            }
            *barrier_flag_ptr = ((*barrier_flag_ptr) + 1) % MAX_ALL_REDUCE_MODULES;
            *barrier_flag_counter_ptr = 0;
        }
    }
}

namespace reduce_fusion
{

inline __device__ float warp_reduce_sum(float val)
{
    val += __shfl_xor_sync(~0, val, 16);
    val += __shfl_xor_sync(~0, val, 8);
    val += __shfl_xor_sync(~0, val, 4);
    val += __shfl_xor_sync(~0, val, 2);
    val += __shfl_xor_sync(~0, val, 1);
    return val;
}

inline __device__ float block_reduce_sum(float val)
{
    __shared__ float smem[details::kWarpSize];
    int lane_id = threadIdx.x % details::kWarpSize, warp_id = threadIdx.x / details::kWarpSize,
        warp_num = blockDim.x / details::kWarpSize;
    val = warp_reduce_sum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();
    val = lane_id < warp_num ? smem[lane_id] : 0.f;
    val = warp_reduce_sum(val);
    return val;
}

template <typename T, typename PackedStruct>
inline __device__ float accumulate(float acc, PackedStruct& vec)
{
    static constexpr int kLoopNum = sizeof(PackedStruct) / sizeof(T);
#pragma unroll
    for (int i = 0; i < kLoopNum; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<T*>(vec.unpacked)[i]);
        acc += v * v;
    }
    return acc;
}

template <typename T, bool Affine, typename PackedStruct>
inline __device__ int4 rms_norm(float denom, PackedStruct& vec, PackedStruct& weight)
{
    static constexpr int kLoopNum = sizeof(PackedStruct) / sizeof(T);
    PackedStruct ret;
#pragma unroll
    for (int i = 0; i < kLoopNum; ++i)
    {
        float v1 = static_cast<float>(reinterpret_cast<T*>(vec.unpacked)[i]);
        if constexpr (Affine)
        {
            float v2 = static_cast<float>(reinterpret_cast<T*>(weight.unpacked)[i]);
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(v1 * denom * v2);
        }
        else
        {
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(v1 * denom);
        }
    }
    return ret.packed;
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false, bool UseSmem = false>
__global__ void rms_norm_kernel(AllReduceParams params)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    extern __shared__ uint8_t smem_ptr[];
    T* smem = reinterpret_cast<T*>(smem_ptr);

    int bid = blockIdx.x, tid = threadIdx.x;

    T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

    int block_offset = bid * params.fusion_params.hidden_size;
    int thread_offset = tid * kPackedSize;

    if constexpr (Residual)
    {
        residual_buffer += block_offset;
    }
    local_final_output_buffer += block_offset;
    intermediate_buffer += block_offset;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    PackedStruct inter_vec, weight_vec;
    float acc = 0.f;
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        inter_vec.packed = *reinterpret_cast<int4 const*>(intermediate_buffer + offset);
        if constexpr (Bias)
        {
            PackedStruct bias_vec;
            bias_vec.packed = *reinterpret_cast<int4 const*>(bias_buffer + offset);
            inter_vec.packed = add128b(inter_vec, bias_vec);
        }
        if constexpr (Residual)
        {
            PackedStruct residual_vec;
            residual_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer + offset);
            inter_vec.packed = add128b(inter_vec, residual_vec);
            *reinterpret_cast<int4*>(intermediate_buffer + offset) = inter_vec.packed;
        }
        acc = accumulate<T>(acc, inter_vec);
        if constexpr (UseSmem)
        {
            *reinterpret_cast<int4*>(&smem[offset]) = inter_vec.packed;
        }
    }
    acc = block_reduce_sum(acc);
    float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        if constexpr (UseSmem)
        {
            inter_vec.packed = *reinterpret_cast<int4 const*>(&smem[offset]);
        }
        if constexpr (Affine)
        {
            weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + offset);
        }
        inter_vec.packed = rms_norm<T, Affine>(denom, inter_vec, weight_vec);
        *reinterpret_cast<int4*>(&local_final_output_buffer[offset]) = inter_vec.packed;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false>
__global__ void rms_pre_post_norm_kernel(AllReduceParams params) // for gemma2 pre residual + post residual norm
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    int bid = blockIdx.x, tid = threadIdx.x;

    T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
    T const* weight_buffer_pre_residual_norm
        = reinterpret_cast<T const*>(params.fusion_params.weight_buffer_pre_residual_norm);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

    int block_offset = bid * params.fusion_params.hidden_size;
    int thread_offset = tid * kPackedSize;

    if constexpr (Residual)
    {
        residual_buffer += block_offset;
    }
    local_final_output_buffer += block_offset;
    intermediate_buffer += block_offset;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    PackedStruct inter_vec, weight_vec, weight_vec_pre_residual_norm, bias_vec;
    float acc = 0.f;
    float acc_pre_residual_norm = 0.f;
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        inter_vec.packed = *reinterpret_cast<int4 const*>(intermediate_buffer + offset);
        if constexpr (Bias)
        {
            bias_vec.packed = *reinterpret_cast<int4 const*>(bias_buffer + offset);
        }

        if constexpr (Bias)
        {
            inter_vec.packed = add128b(inter_vec, bias_vec);
        }

        // pre-residual norm.
        acc_pre_residual_norm = accumulate<T>(acc_pre_residual_norm, inter_vec);
        acc_pre_residual_norm = block_reduce_sum(acc_pre_residual_norm);
        float denom_pre_residual_norm
            = rsqrtf(acc_pre_residual_norm / params.fusion_params.hidden_size + params.fusion_params.eps);

        if constexpr (Affine)
        {
            weight_vec_pre_residual_norm.packed
                = *reinterpret_cast<int4 const*>(weight_buffer_pre_residual_norm + thread_offset);
        }
        inter_vec.packed = rms_norm<T, Affine>(denom_pre_residual_norm, inter_vec, weight_vec_pre_residual_norm);

        if constexpr (Residual)
        {
            PackedStruct residual_vec;
            residual_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer + offset);
            inter_vec.packed = add128b(inter_vec, residual_vec);
            *reinterpret_cast<int4*>(intermediate_buffer + offset) = inter_vec.packed;
        }
        acc = accumulate<T>(acc, inter_vec);
    }
    acc = block_reduce_sum(acc);
    float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        if constexpr (Affine)
        {
            weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + offset);
        }
        inter_vec.packed = rms_norm<T, Affine>(denom, inter_vec, weight_vec);
        *reinterpret_cast<int4*>(&local_final_output_buffer[offset]) = inter_vec.packed;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false>
void rms_norm_kernel_launcher(AllReduceParams& params, cudaStream_t stream, AllReduceFusionOp fusionOp)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    TLLM_CHECK(params.fusion_params.hidden_size % kPackedSize == 0);
    if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
    {
        TLLM_CHECK(params.fusion_params.hidden_size <= 8192);
    }
    int need_threads = params.fusion_params.hidden_size / kPackedSize;
    int cta_size;
    if (need_threads <= details::kMaxCtaSize)
    {
        cta_size = (need_threads + details::kWarpSize - 1) / details::kWarpSize * details::kWarpSize;
    }
    else
    {
        cta_size = details::kMaxCtaSize;
    }
    int cta_num = params.elts_total / params.fusion_params.hidden_size;
    int smem_size = 0;
    if (cta_size * details::kBytesPerAccess / sizeof(T) < params.fusion_params.hidden_size)
    {
        smem_size = params.fusion_params.hidden_size * sizeof(T);
        if (tensorrt_llm::common::getEnvEnablePDL())
        {
            TLLM_LOG_DEBUG("Enable PDL in rms_norm_kernel");
            cudaLaunchConfig_t kernelConfig = {0};
            kernelConfig.gridDim = cta_num;
            kernelConfig.blockDim = cta_size;
            kernelConfig.dynamicSmemBytes = smem_size;
            kernelConfig.stream = stream;

            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            kernelConfig.attrs = attribute;
            kernelConfig.numAttrs = 1;

            if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                TLLM_CUDA_CHECK(
                    cudaLaunchKernelEx(&kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, true>, params));
            }
            else
            { // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
                TLLM_CUDA_CHECK(
                    cudaLaunchKernelEx(&kernelConfig, rms_pre_post_norm_kernel<T, Bias, Residual, Affine>, params));
            }
        }
        else
        {
            if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                rms_norm_kernel<T, Bias, Residual, Affine, true><<<cta_num, cta_size, smem_size, stream>>>(params);
            }
            else
            { // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
                rms_pre_post_norm_kernel<T, Bias, Residual, Affine><<<cta_num, cta_size, smem_size, stream>>>(params);
            }
        }
    }
    else
    {
        if (tensorrt_llm::common::getEnvEnablePDL())
        {
            TLLM_LOG_DEBUG("Enable PDL in rms_norm_kernel");
            cudaLaunchConfig_t kernelConfig = {0};
            kernelConfig.gridDim = cta_num;
            kernelConfig.blockDim = cta_size;
            kernelConfig.dynamicSmemBytes = smem_size;
            kernelConfig.stream = stream;

            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            kernelConfig.attrs = attribute;
            kernelConfig.numAttrs = 1;

            if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                TLLM_CUDA_CHECK(
                    cudaLaunchKernelEx(&kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, false>, params));
            }
            else
            { // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
                TLLM_CUDA_CHECK(
                    cudaLaunchKernelEx(&kernelConfig, rms_pre_post_norm_kernel<T, Bias, Residual, Affine>, params));
            }
        }
        else
        {
            if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                rms_norm_kernel<T, Bias, Residual, Affine, false><<<cta_num, cta_size, smem_size, stream>>>(params);
            }
            else
            { // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
                rms_pre_post_norm_kernel<T, Bias, Residual, Affine><<<cta_num, cta_size, smem_size, stream>>>(params);
            }
        }
    }
}

template <typename T>
struct NegZero128b
{
    static constexpr int v = static_cast<int>(0x80008000);
    static constexpr int4 value = {v, v, v, v};
};

template <>
struct NegZero128b<float>
{
    static constexpr int v = static_cast<int>(0x80000000);
    static constexpr int4 value = {v, v, v, v};
};

template <typename T>
__device__ static constexpr int4 NegZero128b_v = NegZero128b<T>::value;

template <typename T>
__device__ __forceinline__ bool is_neg_zero(T& v);

template <>
__device__ __forceinline__ bool is_neg_zero<float>(float& v)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
    return bits == 0x80000000;
}

template <>
__device__ __forceinline__ bool is_neg_zero<half>(half& v)
{
    uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
    return bits == 0x8000;
}

template <>
__device__ __forceinline__ bool is_neg_zero<__nv_bfloat16>(__nv_bfloat16& v)
{
    uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
    return bits == 0x8000;
}

template <typename ValType, typename VecType>
__device__ __forceinline__ VecType remove_neg_zero(VecType const& vec)
{
    static constexpr int kIter = sizeof(VecType) / sizeof(ValType);
    using ReadOnlyValType = std::add_const_t<ValType>;
    VecType ret;
#pragma unroll
    for (int i = 0; i < kIter; ++i)
    {
        auto val = reinterpret_cast<ReadOnlyValType*>(&vec)[i];
        reinterpret_cast<ValType*>(&ret)[i] = is_neg_zero(val) ? static_cast<ValType>(0.f) : val;
    }
    return ret;
}

template <typename ValType, typename VecType>
__device__ __forceinline__ bool has_neg_zero(VecType const& vec)
{
    static constexpr int kIter = sizeof(VecType) / sizeof(ValType);
    using ReadOnlyValType = std::add_const_t<ValType>;
#pragma unroll
    for (int i = 0; i < kIter; ++i)
    {
        auto val = reinterpret_cast<ReadOnlyValType*>(&vec)[i];
        if (is_neg_zero(val))
        {
            return true;
        }
    }
    return false;
}

template <typename ValType, typename VecType>
__device__ __forceinline__ bool all_neg_zero(VecType const& vec)
{
    static constexpr int kIter = sizeof(VecType) / sizeof(ValType);
    using ReadOnlyValType = std::add_const_t<ValType>;
#pragma unroll
    for (int i = 0; i < kIter; ++i)
    {
        auto val = reinterpret_cast<ReadOnlyValType*>(&vec)[i];
        if (!is_neg_zero(val))
        {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void st_global_release(int4 const& val, int4* addr)
{
    asm volatile("st.release.global.sys.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y), "r"(val.z),
        "r"(val.w), "l"(addr));
}

__device__ __forceinline__ int4 ld_global_acquire(int4* addr)
{
    int4 val;
    asm volatile("ld.acquire.global.sys.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    return val;
}

__device__ __forceinline__ void st_global_volatile(int4 const& val, int4* addr)
{
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w),
        "l"(addr));
}

__device__ __forceinline__ int4 ld_global_volatile(int4* addr)
{
    int4 val;
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    return val;
}

template <typename ValType>
__device__ __forceinline__ void set_neg_zero(int4* addr)
{
    st_global_volatile(NegZero128b_v<ValType>, addr);
}

template <typename T, int RanksPerNode, bool PushMode>
struct Reducer;

template <typename T, int RanksPerNode>
struct Reducer<T, RanksPerNode, true>
{
    static __device__ __forceinline__ int4 allreduce(AllReduceParams& params, int global_offset)
    {
        using PackedStruct = typename PackedOn16Bytes<T>::Type;
        auto const barrier_flag = *params.barrier_flag_ptr;
        int ping = barrier_flag % 3;
        int pong = (barrier_flag + 2) % 3;
        T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
        T* local_shared_buffer = reinterpret_cast<T*>(
            params.fusion_params.lamport_peer_comm_buffer_ptrs[params.local_rank + ping * MAX_RANKS_PER_NODE]);
        T* local_clean_buffer = reinterpret_cast<T*>(
            params.fusion_params.lamport_peer_comm_buffer_ptrs[params.local_rank + pong * MAX_RANKS_PER_NODE]);
        local_input_buffer += global_offset;
        local_shared_buffer += global_offset;
        local_clean_buffer += global_offset;
        T* buffers[RanksPerNode];
#pragma unroll
        for (int ii = 0; ii < RanksPerNode; ++ii)
        {
            int rank = (params.local_rank + ii) % RanksPerNode;
            buffers[ii] = reinterpret_cast<T*>(
                              params.fusion_params.lamport_peer_comm_buffer_ptrs[rank + ping * MAX_RANKS_PER_NODE])
                + global_offset + params.local_rank * params.elts_total;
        }
        PackedStruct sum_vec, val;
        val.packed = remove_neg_zero<T>(*reinterpret_cast<int4 const*>(local_input_buffer));
#pragma unroll
        for (int ii = 1; ii < RanksPerNode; ++ii)
        {
            st_global_volatile(val.packed, reinterpret_cast<int4*>(buffers[ii]));
        }
        sum_vec.packed = val.packed;
#pragma unroll
        for (int ii = 1; ii < RanksPerNode; ++ii)
        {
            int rank = (params.local_rank + ii) % RanksPerNode;
            set_neg_zero<T>(reinterpret_cast<int4*>(local_clean_buffer + rank * params.elts_total));
        }
        PackedStruct vals[RanksPerNode - 1];
        bool done = false;
        while (!done)
        {
            done = true;
#pragma unroll
            for (int ii = 1; ii < RanksPerNode; ++ii)
            {
                int rank = (params.local_rank + ii) % RanksPerNode;
                vals[ii - 1].packed
                    = ld_global_volatile(reinterpret_cast<int4*>(local_shared_buffer + rank * params.elts_total));
            }
#pragma unroll
            for (int ii = 0; ii < RanksPerNode - 1; ii++)
            {
                done &= !has_neg_zero<T>(vals[ii].packed);
            }
        }

#pragma unroll
        for (int ii = 1; ii < RanksPerNode; ++ii)
        {
            sum_vec.packed = add128b(sum_vec, vals[ii - 1]);
        }
        return sum_vec.packed;
    }
};

template <typename T, int RanksPerNode>
struct Reducer<T, RanksPerNode, false>
{
    static __device__ __forceinline__ int4 allreduce(AllReduceParams& params, int global_offset)
    {
        using PackedStruct = typename PackedOn16Bytes<T>::Type;
        auto const barrier_flag = *params.barrier_flag_ptr;
        int ping = barrier_flag % 3;
        int pong = (barrier_flag + 2) % 3;
        T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
        T* local_shared_buffer = reinterpret_cast<T*>(
            params.fusion_params.lamport_peer_comm_buffer_ptrs[params.local_rank + ping * MAX_RANKS_PER_NODE]);
        T* local_clean_buffer = reinterpret_cast<T*>(
            params.fusion_params.lamport_peer_comm_buffer_ptrs[params.local_rank + pong * MAX_RANKS_PER_NODE]);
        local_input_buffer += global_offset;
        local_shared_buffer += global_offset;
        local_clean_buffer += global_offset;
        T* buffers[RanksPerNode];
#pragma unroll
        for (int ii = 0; ii < RanksPerNode; ++ii)
        {
            int rank = (params.local_rank + ii) % RanksPerNode;
            buffers[ii] = reinterpret_cast<T*>(
                              params.fusion_params.lamport_peer_comm_buffer_ptrs[rank + ping * MAX_RANKS_PER_NODE])
                + global_offset;
        }
        PackedStruct sum_vec, val;
        val.packed = remove_neg_zero<T>(*reinterpret_cast<int4 const*>(local_input_buffer));
        st_global_volatile(val.packed, reinterpret_cast<int4*>(local_shared_buffer));
        sum_vec.packed = val.packed;
#pragma unroll
        for (int ii = 1; ii < RanksPerNode; ++ii)
        {
            do
            {
                val.packed = ld_global_volatile(reinterpret_cast<int4*>(buffers[ii]));
            } while (has_neg_zero<T>(val.packed));
            sum_vec.packed = add128b(sum_vec, val);
        }
        set_neg_zero<T>(reinterpret_cast<int4*>(local_clean_buffer));
        return sum_vec.packed;
    }
};

template <int ClusterSize, typename T, int RanksPerNode, bool Bias = false, bool Affine = false, bool PushMode = true>
static __global__ void lamport_style_one_shot_all_reduce_norm_kernel(AllReduceParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    namespace cg = cooperative_groups;
    static_assert(RanksPerNode <= MAX_RANKS_PER_NODE);
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    cg::cluster_group cluster = cg::this_cluster();

    __shared__ float cluster_acc, cluster_acc_sum;

    int bid = blockIdx.x, tid = threadIdx.x;
    int cluster_id = bid / ClusterSize, cluster_block_rank = bid % ClusterSize;

    int token_id = cluster_id;
    int cluster_offset = token_id * params.fusion_params.hidden_size;
    int block_offset = cluster_block_rank * params.fusion_params.hidden_size / ClusterSize;
    int thread_offset = tid * kPackedSize;

    int inner_token_offset = block_offset + thread_offset;
    int global_offset = cluster_offset + inner_token_offset;

    T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

    local_final_output_buffer += global_offset;
    intermediate_buffer += global_offset;
    residual_buffer += global_offset;
    bias_buffer += inner_token_offset;
    weight_buffer += inner_token_offset;

    PackedStruct weight_vec, bias_vec, residual_vec;
    residual_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer);
    if constexpr (Bias)
    {
        bias_vec.packed = *reinterpret_cast<int4 const*>(bias_buffer);
    }
    if constexpr (Affine)
    {
        weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer);
    }

    cudaGridDependencySynchronize();

    float acc = 0.f;
    PackedStruct sum_vec;
    sum_vec.packed = Reducer<T, RanksPerNode, PushMode>::allreduce(params, global_offset);

    if constexpr (Bias)
    {
        sum_vec.packed = add128b(sum_vec, bias_vec);
    }
    sum_vec.packed = add128b(sum_vec, residual_vec);
    *reinterpret_cast<int4*>(intermediate_buffer) = sum_vec.packed;
    acc = accumulate<T>(acc, sum_vec);
    acc = block_reduce_sum(acc);
    if (ClusterSize > 1)
    {
        if (threadIdx.x == 0)
        {
            cluster_acc = acc;
        }
        cluster.sync();
        if (threadIdx.x == 0)
        {
            acc = 0.f;
#pragma unroll
            for (int ii = 0; ii < ClusterSize; ++ii)
            {
                acc += *cluster.map_shared_rank(&cluster_acc, ii);
            }
            cluster_acc_sum = acc;
        }
        __syncthreads();
        acc = cluster_acc_sum;
        cluster.sync();
    }

    float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
    sum_vec.packed = rms_norm<T, Affine>(denom, sum_vec, weight_vec);
    *reinterpret_cast<int4*>(local_final_output_buffer) = sum_vec.packed;
    cudaTriggerProgrammaticLaunchCompletion();
    update_barrier_flag(params.barrier_flag_ptr, params.barrier_flag_counter_ptr);
#endif
}

int heuristic_min_warp_number(int tp_size, int hidden_size)
{
    if (hidden_size >= 4096)
    {
        return 4;
    }
    if (tp_size == 2)
    {
        return 32;
    }
    else
    {
        return 16;
    }
}

template <typename T, int RanksPerNode, bool Bias, bool Affine>
void lamport_style_one_shot_all_reduce_norm_kernel_launcher(AllReduceParams params, cudaStream_t stream)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    TLLM_CHECK(params.fusion_params.hidden_size % kPackedSize == 0);
    int threads_per_token = params.fusion_params.hidden_size / kPackedSize;
    int warps_per_token = (threads_per_token + details::kWarpSize - 1) / details::kWarpSize;
    int token_num = params.elts_total / params.fusion_params.hidden_size;
    int warp_min_number = heuristic_min_warp_number(RanksPerNode, params.fusion_params.hidden_size);
    int cluster_size = std::min(((warps_per_token + warp_min_number - 1) / warp_min_number), details::kClusterMaxSize);
    int cta_size = warps_per_token / cluster_size * details::kWarpSize;
    TLLM_CHECK(cta_size <= details::kMaxCtaSize);
    int cta_num = token_num * cluster_size;
    cudaLaunchConfig_t kernel_config = {0};
    kernel_config.gridDim = cta_num;
    kernel_config.blockDim = cta_size;
    kernel_config.dynamicSmemBytes = 0;
    kernel_config.stream = stream;

    cudaLaunchAttribute attribute[2];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    kernel_config.attrs = attribute;
    kernel_config.numAttrs = 1;
    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        attribute[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[1].val.programmaticStreamSerializationAllowed = 1;
        kernel_config.numAttrs++;
    }
#define LAUNCH_LAMPORT_KERNEL(CLUSTER_SIZE)                                                                            \
    if (cluster_size == CLUSTER_SIZE)                                                                                  \
    {                                                                                                                  \
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernel_config,                                                             \
            lamport_style_one_shot_all_reduce_norm_kernel<CLUSTER_SIZE, T, RanksPerNode, Bias, Affine>, params));      \
        return;                                                                                                        \
    }
    LAUNCH_LAMPORT_KERNEL(1);
    LAUNCH_LAMPORT_KERNEL(2);
    LAUNCH_LAMPORT_KERNEL(3);
    LAUNCH_LAMPORT_KERNEL(4);
    LAUNCH_LAMPORT_KERNEL(5);
    LAUNCH_LAMPORT_KERNEL(6);
    LAUNCH_LAMPORT_KERNEL(7);
    LAUNCH_LAMPORT_KERNEL(8);
#undef LAUNCH_LAMPORT_KERNEL
}

template <typename T, int RanksPerNode, bool Bias = false, bool Affine = false, bool UseSmem = false>
static __global__ void __launch_bounds__(1024, 1) one_shot_all_reduce_norm_kernel(AllReduceParams params)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    extern __shared__ uint8_t smem_ptr[];
    T* smem = reinterpret_cast<T*>(smem_ptr);

    int bid = blockIdx.x, tid = threadIdx.x;
    int norm_num = params.elts_total / params.fusion_params.hidden_size;
    int norm_per_block = (norm_num + gridDim.x - 1) / gridDim.x;
    int norm_this_block = std::min(norm_per_block, norm_num - bid * norm_per_block);

    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

    int block_offset = bid * norm_per_block * params.fusion_params.hidden_size;
    int thread_offset = tid * kPackedSize;

    local_input_buffer += block_offset;
    residual_buffer += block_offset;
    local_final_output_buffer += block_offset;
    intermediate_buffer += block_offset;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    auto const barrier_flag = *params.barrier_flag_ptr;
    auto const buffer_offset = (barrier_flag % 2 == 0) ? 0 : params.ranks_per_node;
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank + buffer_offset]);

    T* buffers[RanksPerNode];
#pragma unroll
    for (int ii = 0; ii < RanksPerNode; ++ii)
    {
        int rank = (params.local_rank + ii) % RanksPerNode;
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank + buffer_offset]);
    }
    local_shared_buffer += block_offset;

    for (int offset = thread_offset; offset < norm_this_block * params.fusion_params.hidden_size;
         offset += blockDim.x * kPackedSize)
    {
        *reinterpret_cast<int4*>(&local_shared_buffer[offset])
            = *reinterpret_cast<int4 const*>(&local_input_buffer[offset]);
    }
    block_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RanksPerNode, tid, bid);
    for (int norm_idx = 0; norm_idx < norm_this_block; ++norm_idx)
    {
        int norm_offset = norm_idx * params.fusion_params.hidden_size;
        float acc = 0.f;
        PackedStruct sum_vec, weight_vec, bias_vec, residual_vec;
        for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
        {
            PackedStruct vals[RanksPerNode];
            sum_vec.packed = {0, 0, 0, 0};
            if constexpr (Bias)
            {
                bias_vec.packed = *reinterpret_cast<int4 const*>(&bias_buffer[offset]);
            }
            residual_vec.packed = *reinterpret_cast<int4 const*>(&residual_buffer[norm_offset + offset]);
#pragma unroll
            for (int ii = 0; ii < RanksPerNode; ++ii)
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][block_offset + norm_offset + offset]);
            }
#pragma unroll
            for (int ii = 0; ii < RanksPerNode; ++ii)
            {
                sum_vec.packed = add128b(sum_vec, vals[ii]);
            }
            if constexpr (Bias)
            {
                sum_vec.packed = add128b(sum_vec, bias_vec);
            }
            sum_vec.packed = add128b(sum_vec, residual_vec);
            *reinterpret_cast<int4*>(&intermediate_buffer[norm_offset + offset]) = sum_vec.packed;
            acc = accumulate<T>(acc, sum_vec);
            if constexpr (UseSmem)
            {
                *reinterpret_cast<int4*>(&smem[offset]) = sum_vec.packed;
            }
        }
        acc = block_reduce_sum(acc);
        float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
        for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
        {
            if constexpr (UseSmem)
            {
                sum_vec.packed = *reinterpret_cast<int4 const*>(&smem[offset]);
            }
            if constexpr (Affine)
            {
                weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + offset);
            }
            sum_vec.packed = rms_norm<T, Affine>(denom, sum_vec, weight_vec);
            *reinterpret_cast<int4*>(&local_final_output_buffer[norm_offset + offset]) = sum_vec.packed;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    update_barrier_flag(params.barrier_flag_ptr, params.barrier_flag_counter_ptr);
}

template <typename T, int RanksPerNode, bool Bias = false, bool Affine = false>
static __global__ void __launch_bounds__(1024, 1) one_shot_prenorm_all_reduce_norm_kernel(AllReduceParams params)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    int bid = blockIdx.x, tid = threadIdx.x;
    int norm_num = params.elts_total / params.fusion_params.hidden_size;
    int norm_per_block = (norm_num + gridDim.x - 1) / gridDim.x;
    int norm_this_block = std::min(norm_per_block, norm_num - bid * norm_per_block);

    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
    T const* weight_buffer_pre_residual_norm
        = reinterpret_cast<T const*>(params.fusion_params.weight_buffer_pre_residual_norm);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

    int block_offset = bid * norm_per_block * params.fusion_params.hidden_size;
    int thread_offset = tid * kPackedSize;

    local_input_buffer += block_offset;
    residual_buffer += block_offset;
    local_final_output_buffer += block_offset;
    intermediate_buffer += block_offset;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    auto const barrier_flag = *params.barrier_flag_ptr;
    auto const buffer_offset = (barrier_flag % 2 == 0) ? 0 : params.ranks_per_node;
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank + buffer_offset]);

    T* buffers[RanksPerNode];
#pragma unroll
    for (int ii = 0; ii < RanksPerNode; ++ii)
    {
        int rank = (params.local_rank + ii) % RanksPerNode;
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank + buffer_offset]);
    }
    local_shared_buffer += block_offset;

    for (int offset = thread_offset; offset < norm_this_block * params.fusion_params.hidden_size;
         offset += blockDim.x * kPackedSize)
    {
        *reinterpret_cast<int4*>(&local_shared_buffer[offset])
            = *reinterpret_cast<int4 const*>(&local_input_buffer[offset]);
    }
    block_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RanksPerNode, tid, bid);
    for (int norm_idx = 0; norm_idx < norm_this_block; ++norm_idx)
    {
        int norm_offset = norm_idx * params.fusion_params.hidden_size;
        float acc = 0.f;
        float acc_pre_residual_norm = 0.f;
        PackedStruct sum_vec, weight_vec, bias_vec, residual_vec, weight_vec_pre_residual_norm;
        for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
        {
            PackedStruct vals[RanksPerNode];
            sum_vec.packed = {0, 0, 0, 0};
            if constexpr (Bias)
            {
                bias_vec.packed = *reinterpret_cast<int4 const*>(&bias_buffer[offset]);
            }
            residual_vec.packed = *reinterpret_cast<int4 const*>(&residual_buffer[norm_offset + offset]);
#pragma unroll
            for (int ii = 0; ii < RanksPerNode; ++ii)
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][block_offset + norm_offset + offset]);
            }
#pragma unroll
            for (int ii = 0; ii < RanksPerNode; ++ii)
            {
                sum_vec.packed = add128b(sum_vec, vals[ii]);
            }

            if constexpr (Bias)
            {
                sum_vec.packed = add128b(sum_vec, bias_vec);
            }

            // norm1 is pre-residual norm.
            acc_pre_residual_norm = accumulate<T>(acc_pre_residual_norm, sum_vec);

            acc_pre_residual_norm = block_reduce_sum(acc_pre_residual_norm);

            float denom_pre_residual_norm
                = rsqrtf(acc_pre_residual_norm / params.fusion_params.hidden_size + params.fusion_params.eps);
            if constexpr (Affine)
            {
                weight_vec_pre_residual_norm.packed
                    = *reinterpret_cast<int4 const*>(weight_buffer_pre_residual_norm + thread_offset);
            }
            sum_vec.packed = rms_norm<T, Affine>(denom_pre_residual_norm, sum_vec, weight_vec_pre_residual_norm);

            sum_vec.packed = add128b(sum_vec, residual_vec);
            *reinterpret_cast<int4*>(&intermediate_buffer[norm_offset + offset]) = sum_vec.packed;
            acc = accumulate<T>(acc, sum_vec);
        }
        acc = block_reduce_sum(acc);
        float denom = rsqrtf(acc / params.fusion_params.hidden_size + params.fusion_params.eps);
        if constexpr (Affine)
        {
            weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + thread_offset);
        }
        sum_vec.packed = rms_norm<T, Affine>(denom, sum_vec, weight_vec);
        *reinterpret_cast<int4*>(&local_final_output_buffer[norm_offset + thread_offset]) = sum_vec.packed;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    update_barrier_flag(params.barrier_flag_ptr, params.barrier_flag_counter_ptr);
}

template <typename T>
bool is_lamport_supported(int token_num, int hidden_size)
{
    static char* disableLamportReduceNormFusionChar = std::getenv("DISABLE_LAMPORT_REDUCE_NORM_FUSION");
    bool disableLamportReduceNormFusion
        = (disableLamportReduceNormFusionChar != nullptr) || common::getEnvForceDeterministicAllReduce();
    if (disableLamportReduceNormFusion)
        return false;
    static int sm = tensorrt_llm::common::getSMVersion();
    if (sm < 90 || sm >= 120)
    {
        return false;
    }
    if (!std::is_same_v<T, half> && !std::is_same_v<T, __nv_bfloat16>)
    {
        return false;
    }
    if (token_num > details::kLamportTokenNumThreshold)
    {
        return false;
    }
    if (hidden_size < details::kLamportHiddenSizeThreshold)
    {
        return false;
    }
    return true;
}

bool is_lamport_supported(nvinfer1::DataType dataType, int token_num, int hidden_size)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: return is_lamport_supported<float>(token_num, hidden_size);
    case nvinfer1::DataType::kHALF: return is_lamport_supported<half>(token_num, hidden_size);
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: return is_lamport_supported<__nv_bfloat16>(token_num, hidden_size);
#endif
    default: return false;
    }
}

template <typename T, int RanksPerNode, bool Bias, bool Affine>
void one_shot_all_reduce_norm_kernel_launcher(AllReduceParams& params, cudaStream_t stream, AllReduceFusionOp fusionOp)
{
    int token_num = params.elts_total / params.fusion_params.hidden_size;

    if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
    {
        TLLM_CHECK(params.fusion_params.hidden_size <= 8192);
    }

    if (is_lamport_supported<T>(token_num, params.fusion_params.hidden_size)
        && (fusionOp != AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM))
    {
        lamport_style_one_shot_all_reduce_norm_kernel_launcher<T, RanksPerNode, Bias, Affine>(params, stream);
    }
    else
    {
        static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
        TLLM_CHECK(params.fusion_params.hidden_size % kPackedSize == 0);
        int need_threads = params.fusion_params.hidden_size / kPackedSize;
        int cta_size;
        if (need_threads <= details::kMaxCtaSize)
        {
            cta_size = (need_threads + details::kWarpSize - 1) / details::kWarpSize * details::kWarpSize;
        }
        else
        {
            cta_size = details::kMaxCtaSize;
        }
        int norm_num = params.elts_total / params.fusion_params.hidden_size;
        int cta_num = std::min(norm_num, static_cast<int>(MAX_ALL_REDUCE_BLOCKS));
        int smem_size = 0;

        if (cta_size * kPackedSize < params.fusion_params.hidden_size)
        {
            smem_size = params.fusion_params.hidden_size * sizeof(T);
            if (tensorrt_llm::common::getEnvEnablePDL())
            {
                TLLM_LOG_DEBUG("Enable PDL in one_shot_all_reduce_norm_kernel");

                cudaLaunchConfig_t kernelConfig = {0};
                kernelConfig.gridDim = cta_num;
                kernelConfig.blockDim = cta_size;
                kernelConfig.dynamicSmemBytes = smem_size;
                kernelConfig.stream = stream;

                cudaLaunchAttribute attribute[1];
                attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                attribute[0].val.programmaticStreamSerializationAllowed = 1;
                kernelConfig.attrs = attribute;
                kernelConfig.numAttrs = 1;
                if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
                {
                    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                        &kernelConfig, one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, true>, params));
                }
                else
                { // fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
                    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                        &kernelConfig, one_shot_prenorm_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine>, params));
                }
            }
            else
            {
                if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
                {
                    one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, true>
                        <<<cta_num, cta_size, smem_size, stream>>>(params);
                }
                else
                {
                    one_shot_prenorm_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine>
                        <<<cta_num, cta_size, smem_size, stream>>>(params);
                }
            }
        }
        else
        {
            if (tensorrt_llm::common::getEnvEnablePDL())
            {
                cudaLaunchConfig_t kernelConfig = {0};
                kernelConfig.gridDim = cta_num;
                kernelConfig.blockDim = cta_size;
                kernelConfig.dynamicSmemBytes = smem_size;
                kernelConfig.stream = stream;

                cudaLaunchAttribute attribute[1];
                attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                attribute[0].val.programmaticStreamSerializationAllowed = 1;
                kernelConfig.attrs = attribute;
                kernelConfig.numAttrs = 1;

                TLLM_LOG_DEBUG("Enable PDL in one_shot_all_reduce_norm_kernel");
                if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
                {
                    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                        &kernelConfig, one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, false>, params));
                }
                else
                { // fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
                    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                        &kernelConfig, one_shot_prenorm_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine>, params));
                }
            }
            else
            {
                if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
                {
                    one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, false>
                        <<<cta_num, cta_size, smem_size, stream>>>(params);
                }
                else
                {
                    one_shot_prenorm_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine>
                        <<<cta_num, cta_size, smem_size, stream>>>(params);
                }
            }
        }
    }
}

template <typename T>
__global__ void lamport_initialize_kernel(T* buffer, size_t size)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;
    for (size_t offset = (blockIdx.x * blockDim.x + threadIdx.x) * kPackedSize; offset < size;
         offset += gridDim.x * blockDim.x * kPackedSize)
    {
        set_neg_zero<T>(reinterpret_cast<int4*>(&buffer[offset]));
    }
}

template <typename T>
void lamport_initialize_kernel_launcher(void* buffer, size_t size, cudaStream_t stream)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    int block_size = 1024;
    int grid_size = (size + 1024 * kPackedSize - 1) / (1024 * kPackedSize);
    lamport_initialize_kernel<T><<<grid_size, block_size, 0, stream>>>(reinterpret_cast<T*>(buffer), size);
}
}; // namespace reduce_fusion

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false>
static __global__ void oneShotAllReduceKernel(AllReduceParams params)
{
    // Suppose that two GPUs participate in the AR exchange, and we start four blocks.
    // The message is partitioned into chunks as detailed below:
    //               message
    //       |-------------------|
    // GPU 0 | B0 | B1 | B2 | B3 |
    // GPU 1 | B0 | B1 | B2 | B3 |
    //
    // Here the step-by-step behavior of one block:
    // 1. B0 copies the chunk it  is responsible for, from local_input to shareable buffer
    // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier)
    // 3. B0 on GPU 0 pull and sum the chunk from GPU 1, writes the result to local_output
    //
    // With PUSH_MODE, we consider that the shared buffer is of size:
    // params.peer_comm_buffer_ptrs: [world_size * 2, world_size, message_size]
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    // For each plugin, the buffer is of size: [world_size, world_size, message_size]
    //
    // Here the step-by-step behavior of one block:
    // 1. B0 push the chunk is it responsible for into all other GPUs:
    //    peer_comm_buffer_ptrs[:, local_gpu, B0 slice]
    // 2. block sync so the block is shared by other GPUs
    // 3. Reduce along second dimension peer_comm_buffer_ptrs[local_gpu, :, B0 slice]

    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;
    auto const barrier_flag = *params.barrier_flag_ptr;
    auto const buffer_offset = (barrier_flag % 2 == 0) ? 0 : params.ranks_per_node;

    // The number of elements packed into one for comms
    static constexpr int PACKED_ELTS = 16 / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank + buffer_offset]);
    T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

    // Start and end offsets of the thread
    size_t const chunk_start = bidx * params.elts_per_block + tidx * PACKED_ELTS;
    size_t const chunk_end = std::min((bidx + 1) * params.elts_per_block, params.elts_total);

    T* buffers[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
    {
        // buffers[0] is always the local buffers. Helps load balancing reads.
        int rank = (params.local_rank + ii) % RANKS_PER_NODE;
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank + buffer_offset]);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    // Copy from local buffer to shareable buffer
    for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * PACKED_ELTS)
    {
        if constexpr (PUSH_MODE)
        {
#pragma unroll
            for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
            {
                *reinterpret_cast<int4*>(&buffers[ii][params.local_rank * params.elts_total + iter_offset])
                    = *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
            }
        }
        else
        {
            *reinterpret_cast<int4*>(&local_shared_buffer[iter_offset])
                = *reinterpret_cast<int4 const*>(&local_input_buffer[iter_offset]);
        }
    }

    // wait for equivalent blocks of other GPUs to have copied data to their shareable buffer
    block_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

    // Each block accumulates the values from the different GPUs on the same node.
    for (size_t iter_offset = chunk_start; iter_offset < chunk_end; iter_offset += blockDim.x * PACKED_ELTS)
    {
        // Iterate over the different ranks/devices on the node to load the values.
        PackedStruct vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            if constexpr (PUSH_MODE)
            {
                vals[ii].packed
                    = *reinterpret_cast<int4 const*>(&buffers[params.local_rank][ii * params.elts_total + iter_offset]);
            }
            else
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][iter_offset]);
            }
        }

        // Sum the values from the different ranks.
        PackedStruct sums;
        sums.packed = {0, 0, 0, 0};
#pragma unroll
        for (int rank = 0; rank < RANKS_PER_NODE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int ii = (rank + RANKS_PER_NODE - params.local_rank) % RANKS_PER_NODE;
            sums.packed = add128b(sums, vals[ii]);
        }
        // Store to the destination buffer.
        *reinterpret_cast<int4*>(&local_output_buffer[iter_offset]) = sums.packed;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    update_barrier_flag(params.barrier_flag_ptr, params.barrier_flag_counter_ptr);
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool Bias = false, bool Residual = false>
static __global__ void __launch_bounds__(512, 1) twoShotAllReduceKernel(AllReduceParams params)
{
    // Suppose that two GPUs participate in the AR exchange, and we start two blocks.
    // The message is partitioned into chunks as detailed below:
    //               message
    //       |-------------------|
    //       |--GPU 0--|--GPU 1--| (GPU responsibility parts)
    // GPU 0 | B0 | B1 | B0 | B1 |
    // GPU 1 | B0 | B1 | B0 | B1 |
    //
    // Here the step-by-step behavior of one block:
    // 1. B0 copies all chunks is it responsible for, from local_input to shareable buffer
    // 2. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #0)
    // 3. B0 on GPU 0 gather and sum the B0 chunks from GPU 1, that are in the GPU 0 responsibility
    //    part (the first half of the message, see GPU responsibility row above)
    // 3bis. Likewise, B0 on GPU 1 copies and sum the chunks for GPU 0,
    //       where GPU 1 is responsible: the second half of the message.
    // 4. B0 on GPU 0 and B0 on GPU 1 wait for each other (block_barrier #1)
    // 5. B0 writes result to local_output. It gathers each chunk from its responsible GPU.
    //    For example, here it reads the first chunk from GPU 0 and second chunk from GPU 1.
    //
    // Note that compared to one-shot, one block (CTA) writes multiple input chunks and write multiple output chunks.
    // However, it's only responsible for the summation of a single chunk.
    //
    // With PUSH_MODE, we consider that the shared buffer is of size:
    // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size / world_size]
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    // For each plugin, the buffer is of size: [world_size, world_size, message_size / world_size]
    //
    // Here the step-by-step behavior of one block:
    // 1. B0 push the chunks is it responsible for into the corresponding GPUs:
    //    peer_comm_buffer_ptrs[target_gpu, local_gpu, current B0 slice]
    // 2. block sync so the blocks have been shared by other GPUs
    // 3. Reduce along second dimension peer_comm_buffer_ptrs[local_gpu, :, B0 slice]
    // 4. block barrier (corresponding blocks have finished reduction)
    // 5. pull and write on local buffer, by reading peer_comm_buffer_ptrs[:, 0, B0 slice] (reduction result is
    //    written at index 0 of 2nd dim)

    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;
    // The number of elements packed into one for comms
    static constexpr int PACKED_ELTS = 16 / sizeof(T);
    using PackedType = typename PackedOn16Bytes<T>::Type;

    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

    size_t const chunk_start = bidx * params.elts_per_block + tidx * PACKED_ELTS;
    size_t const chunk_end = min(chunk_start + params.elts_per_block, params.elts_per_rank);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaGridDependencySynchronize();
#endif

    auto const barrier_flag = *params.barrier_flag_ptr;
    auto const buffer_offset = (barrier_flag % 2 == 0) ? 0 : params.ranks_per_node;
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank + buffer_offset]);

    T* buffers[RANKS_PER_NODE];
    int ranks[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
    {
        // A mapping of the ranks to scatter reads as much as possible
        int rank = (params.local_rank + ii) % RANKS_PER_NODE;
        ranks[ii] = rank;
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank + buffer_offset]);
    }

    // Copy all blocks from local buffer to shareable buffer
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS)
    {
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
            if (offset_rank >= params.elts_total)
            {
                continue;
            }

            if constexpr (PUSH_MODE)
            {
                *reinterpret_cast<int4*>(&buffers[ii][params.local_rank * params.elts_per_rank + local_offset])
                    = *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
            }
            else
            {
                *reinterpret_cast<int4*>(&local_shared_buffer[offset_rank])
                    = *reinterpret_cast<int4 const*>(&local_input_buffer[offset_rank]);
            }
        }
    }
    block_barrier(params.peer_barrier_ptrs_in, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

    // Each block accumulates the values from the different GPUs on the same node.
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS)
    {
        size_t const responsible_block_offset = local_offset + params.rank_offset;

        // Iterate over the different ranks/devices on the node to load the values.
        PackedType vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            if constexpr (PUSH_MODE)
            {
                vals[ii].packed
                    = *reinterpret_cast<int4 const*>(&local_shared_buffer[ii * params.elts_per_rank + local_offset]);
            }
            else
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&buffers[ii][responsible_block_offset]);
            }
        }

        // Sum the values from the different ranks.
        PackedType sums;
        sums.packed = {0, 0, 0, 0};
#pragma unroll
        for (int rank = 0; rank < RANKS_PER_NODE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int ii = (rank + RANKS_PER_NODE - params.local_rank) % RANKS_PER_NODE;
            sums.packed = add128b(sums, vals[ii]);
        }

        // Store to the local buffer.
        if constexpr (PUSH_MODE)
        {
            *reinterpret_cast<int4*>(&local_shared_buffer[local_offset]) = sums.packed;
        }
        else
        {
            *reinterpret_cast<int4*>(&local_shared_buffer[responsible_block_offset]) = sums.packed;
        }
    }

    block_barrier(params.peer_barrier_ptrs_out, barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

    // Gather all needed elts from other intra-node ranks
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED_ELTS)
    {
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            // use round-robin gathering from other ranks
            size_t offset_rank = ranks[ii] * params.elts_per_rank + local_offset;
            if (offset_rank >= params.elts_total)
            {
                continue;
            }
            PackedType sums, residual_vec, bias_vec;
            if constexpr (Bias)
            {
                bias_vec.packed
                    = *reinterpret_cast<int4 const*>(reinterpret_cast<T const*>(params.fusion_params.bias_buffer)
                        + offset_rank % params.fusion_params.hidden_size);
            }
            if constexpr (Residual)
            {
                residual_vec.packed = *reinterpret_cast<int4 const*>(
                    reinterpret_cast<T const*>(params.fusion_params.residual_buffer) + offset_rank);
            }
            if constexpr (PUSH_MODE)
            {
                *reinterpret_cast<int4*>(&local_output_buffer[offset_rank])
                    = *reinterpret_cast<int4*>(&buffers[ii][local_offset]);
                sums.packed = *reinterpret_cast<int4*>(&buffers[ii][local_offset]);
            }
            else
            {
                *reinterpret_cast<int4*>(&local_output_buffer[offset_rank])
                    = *reinterpret_cast<int4*>(&buffers[ii][offset_rank]);
                sums.packed = *reinterpret_cast<int4*>(&buffers[ii][offset_rank]);
            }
            if constexpr (Bias)
            {
                sums.packed = add128b(sums, bias_vec);
            }
            if constexpr (Residual)
            {
                sums.packed = add128b(sums, residual_vec);
            }
            *reinterpret_cast<int4*>(&local_output_buffer[offset_rank]) = sums.packed;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    update_barrier_flag(params.barrier_flag_ptr, params.barrier_flag_counter_ptr);
}

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, nvinfer1::DataType type)
{
    size_t elts_per_thread = 16 / common::getDTypeSize(type);
    int const msg_align = (algo == AllReduceStrategyType::TWOSHOT) ? n_ranks * elts_per_thread : elts_per_thread;
    bool supported_algo = (algo == AllReduceStrategyType::ONESHOT || algo == AllReduceStrategyType::TWOSHOT);
    return supported_algo && (msg_size % msg_align == 0);
}

std::tuple<int, int> kernelLaunchConfig(AllReduceStrategyType algo, AllReduceParams& params, size_t elts_per_thread)
{
    int blocks_per_grid = 1, threads_per_block = DEFAULT_BLOCK_SIZE;

    switch (algo)
    {
    case AllReduceStrategyType::ONESHOT:
    {
        TLLM_CHECK(params.elts_total % elts_per_thread == 0);
        size_t const total_threads = roundUp(params.elts_total / elts_per_thread, WARP_SIZE);
        threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
        blocks_per_grid = std::min(static_cast<size_t>(MAX_ALL_REDUCE_BLOCKS), divUp(total_threads, threads_per_block));
        params.elts_per_block = roundUp(divUp(params.elts_total, blocks_per_grid), elts_per_thread);
        break;
    }
    case AllReduceStrategyType::TWOSHOT:
    {
        TLLM_CHECK(params.elts_total % (elts_per_thread * params.ranks_per_node) == 0);
        size_t const total_threads = roundUp(params.elts_total / (elts_per_thread * params.ranks_per_node), WARP_SIZE);

        /*
        threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
        blocks_per_grid = std::min(static_cast<size_t>(MAX_ALL_REDUCE_BLOCKS), divUp(total_threads, threads_per_block));
        */
        while (total_threads % blocks_per_grid != 0 || total_threads / blocks_per_grid > DEFAULT_BLOCK_SIZE)
        {
            blocks_per_grid += 1;
        }

        threads_per_block = total_threads / blocks_per_grid;

        // NOTE: need to adjust here
        if (blocks_per_grid > MAX_ALL_REDUCE_BLOCKS)
        {
            size_t iter_factor = 1;
            while (blocks_per_grid / iter_factor > MAX_ALL_REDUCE_BLOCKS || blocks_per_grid % iter_factor)
            {
                iter_factor += 1;
            }
            blocks_per_grid /= iter_factor;
        }
        params.elts_per_rank = params.elts_total / params.ranks_per_node;
        params.rank_offset = params.local_rank * params.elts_per_rank;
        params.elts_per_block = roundUp(divUp(params.elts_per_rank, blocks_per_grid), elts_per_thread);
        break;
    }
    default: TLLM_THROW("Algorithm not supported here.");
    }

    return std::make_tuple(blocks_per_grid, threads_per_block);
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool Bias = false, bool Affine = false>
void AllReduceNormKernelLaunch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM || fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM),
        "Unsupported AllReduceFusionOp: %d", static_cast<int>(fusionOp));
    if (algo == AllReduceStrategyType::ONESHOT)
    {
        reduce_fusion::one_shot_all_reduce_norm_kernel_launcher<T, RANKS_PER_NODE, Bias, Affine>(
            params, stream, fusionOp);
    }
    else
    {
        size_t elts_per_thread = 16 / sizeof(T);
        auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(algo, params, elts_per_thread);
        auto output_ptr = params.local_output_buffer_ptr;
        params.local_output_buffer_ptr = params.fusion_params.intermediate_buffer;

        if (tensorrt_llm::common::getEnvEnablePDL())
        {
            TLLM_LOG_DEBUG("Enable PDL in twoShotAllReduceKernel");
            cudaLaunchConfig_t kernelConfig = {0};
            kernelConfig.gridDim = blocks_per_grid;
            kernelConfig.blockDim = threads_per_block;
            kernelConfig.dynamicSmemBytes = 0;
            kernelConfig.stream = stream;

            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            kernelConfig.attrs = attribute;
            kernelConfig.numAttrs = 1;

            TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                &kernelConfig, twoShotAllReduceKernel<T, RANKS_PER_NODE, PUSH_MODE, Bias, true>, params));
        }
        else
        {
            twoShotAllReduceKernel<T, RANKS_PER_NODE, PUSH_MODE, Bias, true>
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(params);
        }
        params.local_output_buffer_ptr = output_ptr;
        reduce_fusion::rms_norm_kernel_launcher<T, false, false, Affine>(params, stream, fusionOp);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false>
void AllReduceNormDispatch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    if (params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, true, true>(algo, config, fusionOp, params, stream);
    }
    else if (params.fusion_params.bias_buffer && !params.fusion_params.weight_buffer)
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, true, false>(algo, config, fusionOp, params, stream);
    }
    else if (!params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, false, true>(algo, config, fusionOp, params, stream);
    }
    else
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, false, false>(algo, config, fusionOp, params, stream);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false>
void AllReduceDispatch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    TLLM_CHECK(fusionOp == AllReduceFusionOp::NONE);
    size_t elts_per_thread = 16 / sizeof(T);
    auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(algo, params, elts_per_thread);
    if (algo == AllReduceStrategyType::ONESHOT)
    {
        auto* kernel_instance = &oneShotAllReduceKernel<T, RANKS_PER_NODE, PUSH_MODE>;
        cudaLaunchConfig_t config;
        config.gridDim = blocks_per_grid;
        config.blockDim = threads_per_block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.attrs = attribute;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, kernel_instance, params);
    }
    else
    {
        auto* kernel_instance = &twoShotAllReduceKernel<T, RANKS_PER_NODE, PUSH_MODE>;
        cudaLaunchConfig_t config;
        config.gridDim = blocks_per_grid;
        config.blockDim = threads_per_block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.attrs = attribute;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, kernel_instance, params);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
void AllReduceDispatchMemcpy(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    if (fusionOp == AllReduceFusionOp::NONE)
    {
        TLLM_LOG_DEBUG("AllReduceDispatch enabled");
        AllReduceDispatch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY>(algo, config, fusionOp, params, stream);
    }
    else
    {
        TLLM_LOG_DEBUG("AllReduceNormDispatch enabled");
        AllReduceNormDispatch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY>(algo, config, fusionOp, params, stream);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false>
void AllReduceDispatchPushMode(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config)
        & static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(AllReduceStrategyConfig::USE_MEMCPY))
    {
        TLLM_LOG_DEBUG("USE_MEMCPY is deprecated and has no effect. ");
    }

    if (fusionOp == AllReduceFusionOp::NONE)
    {
        TLLM_LOG_DEBUG("AllReduceDispatch enabled");
        AllReduceDispatch<T, RANKS_PER_NODE, PUSH_MODE>(algo, config, fusionOp, params, stream);
    }
    else
    {
        TLLM_LOG_DEBUG("AllReduceNormDispatch enabled");
        AllReduceNormDispatch<T, RANKS_PER_NODE, PUSH_MODE>(algo, config, fusionOp, params, stream);
    }
}

template <typename T, int RANKS_PER_NODE> //, bool USE_MEMCPY = false, bool PUSH_MODE = false>
void AllReduceDispatchRanksPerNode(AllReduceStrategyType algo, AllReduceStrategyConfig config,
    AllReduceFusionOp fusionOp, AllReduceParams& params, cudaStream_t stream)
{
    if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config)
        & static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(AllReduceStrategyConfig::PUSH_MODE))
    {
        AllReduceDispatchPushMode<T, RANKS_PER_NODE, true>(algo, config, fusionOp, params, stream);
    }
    else
    {
        AllReduceDispatchPushMode<T, RANKS_PER_NODE, false>(algo, config, fusionOp, params, stream);
    }
}

template <typename T>
void AllReduceDispatchType(AllReduceParams& params, AllReduceStrategyType strat, AllReduceStrategyConfig config,
    AllReduceFusionOp fusionOp, cudaStream_t stream)
{
    switch (params.ranks_per_node)
    {
    case 2: AllReduceDispatchRanksPerNode<T, 2>(strat, config, fusionOp, params, stream); break;
    case 4: AllReduceDispatchRanksPerNode<T, 4>(strat, config, fusionOp, params, stream); break;
    case 6: AllReduceDispatchRanksPerNode<T, 6>(strat, config, fusionOp, params, stream); break;
    case 8: AllReduceDispatchRanksPerNode<T, 8>(strat, config, fusionOp, params, stream); break;
    case 16: AllReduceDispatchRanksPerNode<T, 16>(strat, config, fusionOp, params, stream); break;
    default: TLLM_THROW("Custom all reduce only supported on {2, 4, 6, 8, 16} GPUs per node.");
    }
}

AllReduceParams AllReduceParams::deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, nvinfer1::DataType dataType,
    int token_num, int hidden_size, AllReduceFusionOp op)
{
    void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);
    int flag_offset;
    if (op == AllReduceFusionOp::RESIDUAL_RMS_NORM
        && reduce_fusion::is_lamport_supported(dataType, token_num, hidden_size))
    {
        flag_offset = 0;
    }
    else
    {
        flag_offset = 1;
    }
    auto const flag_ptr
        = buffer[tensorrt_llm::utils::customAllReduceUtils::NUM_POINTERS_PER_RANK * tpSize + flag_offset];
    AllReduceParams params;

    for (int i = 0; i < tpSize * 2; ++i)
    {
        params.peer_comm_buffer_ptrs[i] = buffer_ptrs[i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[2 * tpSize + i]);
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[3 * tpSize + i]);
    }
    params.barrier_flag_ptr = reinterpret_cast<uint32_t*>(flag_ptr);
    params.barrier_flag_counter_ptr = reinterpret_cast<uint32_t*>(
        buffer[tensorrt_llm::utils::customAllReduceUtils::NUM_POINTERS_PER_RANK * tpSize + 2]);
    params.ranks_per_node = tpSize;
    params.local_rank = tpRank;

    return params;
}

void customAllReduce(kernels::AllReduceParams& params, nvinfer1::DataType dataType, AllReduceStrategyType strat,
    AllReduceStrategyConfig config, AllReduceFusionOp fusionOp, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(configurationSupported(strat, params.elts_total, params.ranks_per_node, dataType),
        "Custom all-reduce configuration unsupported");

    sync_check_cuda_error(stream);

    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: AllReduceDispatchType<float>(params, strat, config, fusionOp, stream); break;
    case nvinfer1::DataType::kHALF: AllReduceDispatchType<half>(params, strat, config, fusionOp, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        AllReduceDispatchType<__nv_bfloat16>(params, strat, config, fusionOp, stream);
        break;
#endif
    default: TLLM_THROW("Unsupported dataType for customAllReduce");
    }
    sync_check_cuda_error(stream);
}

template <typename T>
void launchResidualRmsNormKernel(kernels::AllReduceParams& params, cudaStream_t stream, AllReduceFusionOp fusionOp)
{
    if (params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        reduce_fusion::rms_norm_kernel_launcher<T, true, true, true>(params, stream, fusionOp);
    }
    else if (params.fusion_params.bias_buffer && !params.fusion_params.weight_buffer)
    {
        reduce_fusion::rms_norm_kernel_launcher<T, true, true, false>(params, stream, fusionOp);
    }
    else if (!params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        reduce_fusion::rms_norm_kernel_launcher<T, false, true, true>(params, stream, fusionOp);
    }
    else
    {
        reduce_fusion::rms_norm_kernel_launcher<T, false, true, false>(params, stream, fusionOp);
    }
}

void residualRmsNorm(
    kernels::AllReduceParams& params, nvinfer1::DataType dataType, cudaStream_t stream, AllReduceFusionOp fusionOp)
{
    sync_check_cuda_error(stream);
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: launchResidualRmsNormKernel<float>(params, stream, fusionOp); break;
    case nvinfer1::DataType::kHALF: launchResidualRmsNormKernel<half>(params, stream, fusionOp); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: launchResidualRmsNormKernel<__nv_bfloat16>(params, stream, fusionOp); break;
#endif
    default: TLLM_THROW("Unsupported dataType for customAllReduce");
    }
    sync_check_cuda_error(stream);
}

void lamportInitialize(void* buffer, size_t size, nvinfer1::DataType dataType, cudaStream_t stream)
{
    sync_check_cuda_error(stream);
    if (size == 0)
    {
        return;
    }
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        reduce_fusion::lamport_initialize_kernel_launcher<float>(buffer, size, stream);
        break;
    case nvinfer1::DataType::kHALF:
        reduce_fusion::lamport_initialize_kernel_launcher<half>(buffer, size, stream);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        reduce_fusion::lamport_initialize_kernel_launcher<__nv_bfloat16>(buffer, size, stream);
        break;
#endif
    default: TLLM_THROW("Unsupported dataType for customAllReduce");
    }
    sync_check_cuda_error(stream);
}

} // namespace tensorrt_llm::kernels
