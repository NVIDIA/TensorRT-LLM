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
#include "tensorrt_llm/kernels/communicationKernels/allReduceUtils.h"
#include <cooperative_groups.h>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace tensorrt_llm::kernels
{

using tensorrt_llm::common::divUp;
using tensorrt_llm::common::roundUp;

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
    cudaLaunchConfig_t kernelConfig = {0};
    cudaLaunchAttribute attribute[1];
    kernelConfig.gridDim = cta_num;
    kernelConfig.blockDim = cta_size;
    kernelConfig.dynamicSmemBytes = 0;
    kernelConfig.numAttrs = 0;
    kernelConfig.attrs = nullptr;
    kernelConfig.stream = stream;
    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        TLLM_LOG_DEBUG("Enable PDL in rms_norm_kernel");
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        kernelConfig.attrs = attribute;
        kernelConfig.numAttrs = 1;
    }
    if (fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
    {
        if (cta_size * details::kBytesPerAccess / sizeof(T) < params.fusion_params.hidden_size)
        {
            kernelConfig.dynamicSmemBytes = params.fusion_params.hidden_size * sizeof(T);
            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, true>, params));
        }
        else
        {
            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, false>, params));
        }
    }
    else
    { // AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig, rms_pre_post_norm_kernel<T, Bias, Residual, Affine>, params));
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

    tensorrt_llm::kernels::ar_fusion::SyncComm<RanksPerNode> comm(params.workspace);
    T* local_shared_buffer = reinterpret_cast<T*>(comm.comm_bufs[params.local_rank]);

    T* buffers[RanksPerNode];
#pragma unroll
    for (int ii = 0; ii < RanksPerNode; ++ii)
    {
        int rank = (params.local_rank + ii) % RanksPerNode;
        buffers[ii] = reinterpret_cast<T*>(comm.comm_bufs[rank]);
    }
    local_shared_buffer += block_offset;

    for (int offset = thread_offset; offset < norm_this_block * params.fusion_params.hidden_size;
         offset += blockDim.x * kPackedSize)
    {
        *reinterpret_cast<int4*>(&local_shared_buffer[offset])
            = *reinterpret_cast<int4 const*>(&local_input_buffer[offset]);
    }
    tensorrt_llm::kernels::ar_fusion::Barrier<RanksPerNode> barrier(params.local_rank, comm);
    barrier.sync();
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
}
}; // namespace reduce_fusion

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, nvinfer1::DataType type)
{
    size_t elts_per_thread = 16 / common::getDTypeSize(type);
    int const msg_align = (algo == AllReduceStrategyType::TWOSHOT) ? n_ranks * elts_per_thread : elts_per_thread;
    bool supported_algo = (algo == AllReduceStrategyType::ONESHOT || algo == AllReduceStrategyType::TWOSHOT);
    return supported_algo && (msg_size % msg_align == 0);
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool Bias = false, bool Affine = false>
void AllReduceNormKernelLaunch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM, "Unsupported AllReduceFusionOp: %d",
        static_cast<int>(fusionOp));

    TLLM_CHECK(params.fusion_params.hidden_size <= 8192);

    static constexpr int kPackedSize = reduce_fusion::details::kBytesPerAccess / sizeof(T);
    TLLM_CHECK(params.fusion_params.hidden_size % kPackedSize == 0);
    int need_threads = params.fusion_params.hidden_size / kPackedSize;
    int cta_size;
    if (need_threads <= reduce_fusion::details::kMaxCtaSize)
    {
        cta_size = (need_threads + reduce_fusion::details::kWarpSize - 1) / reduce_fusion::details::kWarpSize
            * reduce_fusion::details::kWarpSize;
    }
    else
    {
        cta_size = reduce_fusion::details::kMaxCtaSize;
    }
    int norm_num = params.elts_total / params.fusion_params.hidden_size;
    int cta_num = std::min(norm_num, static_cast<int>(MAX_ALL_REDUCE_BLOCKS));
    cudaLaunchConfig_t kernelConfig = {0};
    cudaLaunchAttribute attribute[1];
    kernelConfig.gridDim = cta_num;
    kernelConfig.blockDim = cta_size;
    kernelConfig.dynamicSmemBytes = 0;
    kernelConfig.stream = stream;
    kernelConfig.numAttrs = 0;
    kernelConfig.attrs = nullptr;
    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        TLLM_LOG_DEBUG("Enable PDL in one_shot_all_reduce_norm_kernel");
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        kernelConfig.attrs = attribute;
        kernelConfig.numAttrs = 1;
    }

    TLLM_CHECK(fusionOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM);

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig,
        reduce_fusion::one_shot_prenorm_all_reduce_norm_kernel<T, RANKS_PER_NODE, Bias, Affine>, params));
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
void AllReduceDispatchPushMode(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    if (static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(config)
        & static_cast<std::underlying_type_t<AllReduceStrategyConfig>>(AllReduceStrategyConfig::USE_MEMCPY))
    {
        TLLM_LOG_DEBUG("USE_MEMCPY is deprecated and has no effect. ");
    }

    TLLM_LOG_DEBUG("AllReduceNormDispatch enabled");
    AllReduceNormDispatch<T, RANKS_PER_NODE, PUSH_MODE>(algo, config, fusionOp, params, stream);
}

template <typename T, int RANKS_PER_NODE> //, bool PUSH_MODE = false>
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

AllReduceParams AllReduceParams::deserialize(int64_t* buffer, size_t tpSize, size_t tpRank)
{
    AllReduceParams params;

    params.workspace = reinterpret_cast<void**>(buffer);
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
} // namespace tensorrt_llm::kernels
