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
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
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
    size_t const world_size, int const tidx, int const bidx, int const grid_size)
{
    // After this function, the block of id == bidx of each GPU has reached the barrier
    if (tidx < world_size)
    {
        // we can think of signals having the shape [world_size, 2, num_blocks, world_size]
        // (+ an offset on dim 2 to account for flags used in multi_gpu_barrier)
        // Dimension 0 is the "listening" dimension, dimension 3 is "emitting" dimension

        // Block broadcast its flag (local_rank on emitting dimension) to all receivers
        uint32_t flag_block_offset = world_size + bidx * world_size;

        if (flag % 2 == 1)
        {
            flag_block_offset += (grid_size + 1) * world_size;
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

namespace reduce_fusion
{
namespace details
{
static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
}; // namespace details

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
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(__fdividef(v1, denom) * v2);
        }
        else
        {
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(__fdividef(v1, denom));
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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
    float denom = __fsqrt_rn(__fdividef(acc, params.fusion_params.hidden_size) + params.fusion_params.eps);
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false>
void rms_norm_kernel_launcher(AllReduceParams params, cudaStream_t stream)
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

            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, true>, params));
        }
        else
        {
            rms_norm_kernel<T, Bias, Residual, Affine, true><<<cta_num, cta_size, smem_size, stream>>>(params);
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

            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&kernelConfig, rms_norm_kernel<T, Bias, Residual, Affine, false>, params));
        }
        else
        {
            rms_norm_kernel<T, Bias, Residual, Affine, false><<<cta_num, cta_size, smem_size, stream>>>(params);
        }
    }
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
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T* intermediate_buffer = reinterpret_cast<T*>(params.fusion_params.intermediate_buffer);

    int block_offset = bid * norm_per_block * params.fusion_params.hidden_size;
    int thread_offset = tid * kPackedSize;

    local_input_buffer += block_offset;
    residual_buffer += block_offset;
    local_shared_buffer += block_offset;
    local_final_output_buffer += block_offset;
    intermediate_buffer += block_offset;

    T* buffers[RanksPerNode];
#pragma unroll
    for (int ii = 0; ii < RanksPerNode; ++ii)
    {
        int rank = (params.local_rank + ii) % RanksPerNode;
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (int offset = thread_offset; offset < norm_this_block * params.fusion_params.hidden_size;
         offset += blockDim.x * kPackedSize)
    {
        *reinterpret_cast<int4*>(&local_shared_buffer[offset])
            = *reinterpret_cast<int4 const*>(&local_input_buffer[offset]);
    }
    block_barrier(
        params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RanksPerNode, tid, bid, gridDim.x);
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
        float denom = __fsqrt_rn(__fdividef(acc, params.fusion_params.hidden_size) + params.fusion_params.eps);
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, int RanksPerNode, bool Bias, bool Affine>
void one_shot_all_reduce_norm_kernel_launcher(AllReduceParams params, cudaStream_t stream)
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

            TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                &kernelConfig, one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, true>, params));
        }
        else
        {
            one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, true>
                <<<cta_num, cta_size, smem_size, stream>>>(params);
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
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                &kernelConfig, one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, false>, params));
        }
        else
        {
            one_shot_all_reduce_norm_kernel<T, RanksPerNode, Bias, Affine, false>
                <<<cta_num, cta_size, smem_size, stream>>>(params);
        }
    }
}
}; // namespace reduce_fusion

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true, bool PUSH_MODE = false>
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
    // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during step 2.
    // We only to know if the other GPU as arrived at the AR kernel, that would mean that data is ready
    //
    // With PUSH_MODE, we consider that the shared buffer is of size:
    // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size]
    //
    // Here the step-by-step behavior of one block:
    // 1. B0 push the chunk is it responsible for into all other GPUs:
    //    params.peer_comm_buffer_ptrs[:, local_gpu, B0 slice]
    // 2. block sync so the block is shared by other GPUs
    // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]

    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;
    int const grid_size = gridDim.x;

    // The number of elements packed into one for comms
    static constexpr int PACKED_ELTS = 16 / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
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
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
    }

    if constexpr (PUSH_MODE || COPY_INPUT)
    {
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
        block_barrier(
            params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
    }
    else
    {
        // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
        multi_gpu_barrier(
            params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);
    }

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
}

template <typename T, int RANKS_PER_NODE, bool COPY_INPUT = true, bool PUSH_MODE = false, bool Bias = false,
    bool Residual = false>
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
    // With COPY_INPUT == false, skip step 1. and use gpu_barrier instead of block barrier during step 2.
    // We only to know if the other GPU as arrived at the AR kernel, that would mean that data is ready
    // to be read.
    //
    // Note that compared to one-shot, one block (CTA) writes multiple input chunks and write multiple output chunks.
    // However, it's only responsible for the summation of a single chunk.
    //
    // With PUSH_MODE, we consider that the shared buffer is of size:
    // params.peer_comm_buffer_ptrs: [world_size, world_size, message_size / world_size]
    //
    // Here the step-by-step behavior of one block:
    // 1. B0 push the chunks is it responsible for into the corresponding GPUs:
    //    params.peer_comm_buffer_ptrs[target_gpu, local_gpu, current B0 slice]
    // 2. block sync so the blocks have been shared by other GPUs
    // 3. Reduce along second dimension params.peer_comm_buffer_ptrs[local_gpu, :, B0 slice]
    // 4. block barrier (corresponding blocks have finished reduction)
    // 5. pull and write on local buffer, by reading params.peer_comm_buffer_ptrs[:, 0, B0 slice] (reduction result is
    //    written at index 0 of 2nd dim)

    int const bidx = blockIdx.x;
    int const tidx = threadIdx.x;
    int const grid_size = gridDim.x;

    // The number of elements packed into one for comms
    static constexpr int PACKED_ELTS = 16 / sizeof(T);
    using PackedType = typename PackedOn16Bytes<T>::Type;

    T const* local_input_buffer = reinterpret_cast<T const*>(params.local_input_buffer_ptr);
    T* local_shared_buffer = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[params.local_rank]);
    T* local_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);

    size_t const chunk_start = bidx * params.elts_per_block + tidx * PACKED_ELTS;
    size_t const chunk_end = min(chunk_start + params.elts_per_block, params.elts_per_rank);

    T* buffers[RANKS_PER_NODE];
    int ranks[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
    {
        // A mapping of the ranks to scatter reads as much as possible
        int rank = (params.local_rank + ii) % RANKS_PER_NODE;
        ranks[ii] = rank;
        buffers[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if constexpr (PUSH_MODE || COPY_INPUT)
    {
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
        block_barrier(
            params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);
    }
    else
    {
        // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
        multi_gpu_barrier(
            params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);
    }

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

    block_barrier(
        params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx, grid_size);

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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
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

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false, bool Bias = false,
    bool Affine = false>
void AllReduceNormKernelLaunch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(fusionOp == AllReduceFusionOp::RESIDUAL_RMS_NORM, "Unsupported AllReduceFusionOp: %d",
        static_cast<int>(fusionOp));
    if (algo == AllReduceStrategyType::ONESHOT)
    {
        reduce_fusion::one_shot_all_reduce_norm_kernel_launcher<T, RANKS_PER_NODE, Bias, Affine>(params, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(!(USE_MEMCPY && PUSH_MODE), "Memcpy cannot be used with PUSH_MODE.");
        size_t elts_per_thread = 16 / sizeof(T);
        auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(algo, params, elts_per_thread);
        if (USE_MEMCPY)
        {
            cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], params.local_input_buffer_ptr,
                params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
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
                &kernelConfig, twoShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE, Bias, true>, params));
        }
        else
        {
            twoShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE, Bias, true>
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(params);
        }
        params.local_output_buffer_ptr = output_ptr;
        reduce_fusion::rms_norm_kernel_launcher<T, false, false, Affine>(params, stream);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
void AllReduceNormDispatch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    if (params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, true, true>(
            algo, config, fusionOp, params, stream);
    }
    else if (params.fusion_params.bias_buffer && !params.fusion_params.weight_buffer)
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, true, false>(
            algo, config, fusionOp, params, stream);
    }
    else if (!params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, false, true>(
            algo, config, fusionOp, params, stream);
    }
    else
    {
        AllReduceNormKernelLaunch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY, false, false>(
            algo, config, fusionOp, params, stream);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
void AllReduceDispatch(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    TLLM_CHECK(fusionOp == AllReduceFusionOp::NONE);
    TLLM_CHECK_WITH_INFO(!(USE_MEMCPY && PUSH_MODE), "Memcpy cannot be used with PUSH_MODE.");
    size_t elts_per_thread = 16 / sizeof(T);
    auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(algo, params, elts_per_thread);
    if (USE_MEMCPY)
    {
        cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], params.local_input_buffer_ptr,
            params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }
    if (algo == AllReduceStrategyType::ONESHOT)
    {
        oneShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(params);
    }
    else
    {
        twoShotAllReduceKernel<T, RANKS_PER_NODE, !USE_MEMCPY, PUSH_MODE>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(params);
    }
}

template <typename T, int RANKS_PER_NODE, bool PUSH_MODE = false, bool USE_MEMCPY = false>
void AllReduceDispatchMemcpy(AllReduceStrategyType algo, AllReduceStrategyConfig config, AllReduceFusionOp fusionOp,
    AllReduceParams& params, cudaStream_t stream)
{
    if (fusionOp == AllReduceFusionOp::NONE)
    {
        AllReduceDispatch<T, RANKS_PER_NODE, PUSH_MODE, USE_MEMCPY>(algo, config, fusionOp, params, stream);
    }
    else
    {
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
        AllReduceDispatchMemcpy<T, RANKS_PER_NODE, PUSH_MODE, true>(algo, config, fusionOp, params, stream);
    }
    else
    {
        AllReduceDispatchMemcpy<T, RANKS_PER_NODE, PUSH_MODE, false>(algo, config, fusionOp, params, stream);
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
    default: TLLM_THROW("Custom all reduce only supported on {2, 4, 6, 8} GPUs per node.");
    }
}

AllReduceParams AllReduceParams::deserialize(int64_t* buffer, size_t tpSize, size_t tpRank)
{
    void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);
    auto const flag_ptr = &buffer[4 * tpSize];
    // cannot use 0 since 0 represents released state for barrier
    *flag_ptr += 1;
    TLLM_LOG_TRACE("AllReduceParams's flag value is %d", *flag_ptr);
    uint32_t flag_value = *flag_ptr;
    AllReduceParams params;
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    auto const buffer_offset = (flag_value % 2 == 0) ? 0 : tpSize;

    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_comm_buffer_ptrs[i] = buffer_ptrs[buffer_offset + i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[2 * tpSize + i]);
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[3 * tpSize + i]);
    }
    params.barrier_flag = flag_value;
    params.ranks_per_node = tpSize;
    params.local_rank = tpRank;

    return params;
}

void customAllReduce(kernels::AllReduceParams& params, nvinfer1::DataType dataType, AllReduceStrategyType strat,
    AllReduceStrategyConfig config, AllReduceFusionOp fusionOp, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(configurationSupported(strat, params.elts_total, params.ranks_per_node, dataType),
        "Custom all-reduce configuration unsupported");

    sync_check_cuda_error();

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
    sync_check_cuda_error();
}

template <typename T>
void launchResidualRmsNormKernel(kernels::AllReduceParams& params, cudaStream_t stream)
{
    if (params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        reduce_fusion::rms_norm_kernel_launcher<T, true, true, true>(params, stream);
    }
    else if (params.fusion_params.bias_buffer && !params.fusion_params.weight_buffer)
    {
        reduce_fusion::rms_norm_kernel_launcher<T, true, true, false>(params, stream);
    }
    else if (!params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        reduce_fusion::rms_norm_kernel_launcher<T, false, true, true>(params, stream);
    }
    else
    {
        reduce_fusion::rms_norm_kernel_launcher<T, false, true, false>(params, stream);
    }
}

void residualRmsNorm(kernels::AllReduceParams& params, nvinfer1::DataType dataType, cudaStream_t stream)
{
    sync_check_cuda_error();
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: launchResidualRmsNormKernel<float>(params, stream); break;
    case nvinfer1::DataType::kHALF: launchResidualRmsNormKernel<half>(params, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: launchResidualRmsNormKernel<__nv_bfloat16>(params, stream); break;
#endif
    default: TLLM_THROW("Unsupported dataType for customAllReduce");
    }
    sync_check_cuda_error();
}

} // namespace tensorrt_llm::kernels
