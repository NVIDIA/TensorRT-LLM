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
#include "mnnvlAllreduceKernels.h"
#include <array>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <nvml.h>
#include <tuple>
#include <type_traits>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

namespace tensorrt_llm::kernels::mnnvl
{

using tensorrt_llm::common::isNegZero;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::cuda_cast;
using tensorrt_llm::common::getMultiProcessorCount;
using tensorrt_llm::common::getDTypeSize;

// Guard the helper function used for this kernel.
namespace detail
{
template <typename PackedType, typename T>
union PackedVec
{
    PackedType packed;
    T elements[sizeof(PackedType) / sizeof(T)];

    __device__ PackedVec& operator+=(PackedVec& other)
    {
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            elements[i] += other.elements[i];
        }
        return *this;
    }

    __device__ PackedVec operator+(PackedVec& other)
    {
        PackedVec result;
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            result.elements[i] = elements[i] + other.elements[i];
        }
        return result;
    }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr)
{
    return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr)
{
    return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename PackedType>
inline __device__ PackedType loadPackedVolatile(void const* ptr)
{
    static_assert(sizeof(PackedType) == 0, "Not implemented");
    return PackedType{};
}

template <>
inline __device__ float4 loadPackedVolatile<float4>(void const* ptr)
{
    float4 return_value;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(return_value.x), "=f"(return_value.y), "=f"(return_value.z), "=f"(return_value.w)
                 : "l"(ptr));
    return return_value;
}

template <>
inline __device__ float2 loadPackedVolatile<float2>(void const* ptr)
{
    float2 return_value;
    asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(return_value.x), "=f"(return_value.y) : "l"(ptr));
    return return_value;
}

template <typename T_IN>
inline __device__ void copy_f4(T_IN* dst, T_IN const* src)
{
    float4* dst4 = reinterpret_cast<float4*>(dst);
    float4 const* src4 = reinterpret_cast<float4 const*>(src);
    __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

#define WARP_SIZE 32U
#define LOG2_WARP_SIZE 5U
#define LANE_ID_MASK 0x1f

template <typename T>
inline __device__ T warpReduceSumPartial(T val)
{
    int lane_id = threadIdx.x & LANE_ID_MASK;
    // We make sure only the last warp will call this function
    int warp_size = blockDim.x - (threadIdx.x & ~(WARP_SIZE - 1));
    unsigned int active_mask = (1U << warp_size) - 1;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        int target_lane = lane_id ^ mask;
        auto tmp = __shfl_xor_sync(active_mask, val, mask, WARP_SIZE);
        val += target_lane < warp_size ? tmp : 0;
    }
    return val;
}

// SYNC:
//  - True: share the sume across all threads
//  - False: only thread 0 get the sum; Other thread's value is undefined.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val)
{
    __shared__ T smem[WARP_SIZE + 1];
    int lane_id = threadIdx.x & LANE_ID_MASK;
    int warp_id = threadIdx.x >> LOG2_WARP_SIZE;
    int warp_num = (blockDim.x + WARP_SIZE - 1) >> LOG2_WARP_SIZE; // Ceiling division to include partial warps

    val = (warp_id == warp_num - 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        val = (lane_id < warp_num) ? smem[lane_id] : (T) 0.f;
        // Need to consider the corner case where we only have one warp and it is partial
        val = (warp_num == 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);

        if constexpr (SYNC)
        {
            if (lane_id == 0)
            {
                smem[warp_id] = val;
            }
        }
    }
    if constexpr (SYNC)
    {
        __syncthreads();
        val = smem[0];
    }
    return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val)
{
    bool has_partial_warp = (blockDim.x & LANE_ID_MASK) != 0;
    if (has_partial_warp)
    {
        return blockReduceSumPartial<T, SYNC>(val);
    }
    else
    {
        return tensorrt_llm::common::blockReduceSum<T>(val);
    }
}

#undef WARP_SIZE
#undef LOG2_WARP_SIZE
#undef LANE_ID_MASK

// We have to define this again since the one in mathUtils.h is shadowed by the one from cudaUtils.h, which is a
// host-only function!
template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

// A helper function to tune the grid configuration for fused oneshot and rmsnorm kernels
// Return (block_size, cluster_size, loads_per_thread)
// We'd prefer 4 warps (128 threads) per CTA and prefer to increase the occupancy by using CGA
// The Adjustment logic is as follows:
// 1. Reduce cluster_size if total threads exceed what's needed
// 2. If we have too few CTAs, trying to use CGA to increase occupancy
// 3. Trying to scale up use multiple loads or CGA
// 4. The block_size is adjusted to be the minimum value that satisfies the above conditions
template <bool USE_CGA = false>
inline std::tuple<int, int, int> adjust_grid_config(int num_tokens, int dim, int elts_per_thread)
{
    // Start with preferred block_size and cluster_size
    int cluster_size = USE_CGA ? 8 : 1;
    int block_size = 128;
    // ========================== Adjust the grid configuration ==========================
    int threads_needed = divUp(dim, elts_per_thread);
    int loads_per_thread = 1;

    block_size = divUp(threads_needed, cluster_size);
    if (USE_CGA)
    {
        while (threads_needed % cluster_size != 0 && cluster_size > 1)
        {
            cluster_size /= 2;
        }
        block_size = divUp(threads_needed, cluster_size);
        while (block_size < 128 && cluster_size >= 2)
        {
            block_size *= 2;
            cluster_size /= 2;
        }
        int sm_count = getMultiProcessorCount();
        while (num_tokens * cluster_size > sm_count && cluster_size > 1 && block_size <= 512)
        {
            block_size *= 2;
            cluster_size /= 2;
        }
    }

    // Trying to scale up use multiple loads or CGA
    while (block_size > 1024)
    {
        if constexpr (USE_CGA)
        {
            if (cluster_size < 8)
            {
                cluster_size = cluster_size << 1;
            }
            else
            {
                break;
            }
        }
        else
        {
            if (loads_per_thread < 8)
            {
                loads_per_thread += 1;
            }
            else
            {
                break;
            }
        }
        block_size = divUp(threads_needed, cluster_size * loads_per_thread);
    }
    return {block_size, cluster_size, loads_per_thread};
}

} // namespace detail

using detail::PackedVec;
using detail::loadPacked;
using detail::loadPackedVolatile;
using detail::blockReduceSum;
using detail::divUp;
using detail::copy_f4;

// Use another macro to enhance readability
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define SUPPORT_CGA
#endif

template <uint8_t WORLD_SIZE, typename T, bool RESNORM_FUSION = false, typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshot_allreduce_fusion_kernel(T* output_ptr, T* prenormed_ptr,
    T const* shard_ptr, T const* residual_in_ptr, T const* gamma_ptr, T** input_ptrs, T* mcast_ptr,
    int const num_tokens, int const token_dim, float epsilon, int const rank, uint32_t* buffer_flags)
{
    constexpr int ELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr int LAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
    constexpr uint32_t ELT_SIZE = sizeof(T);
#ifdef SUPPORT_CGA
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    int packed_idx = cluster.thread_rank();
    int token = blockIdx.x;
    int thread_offset = token * token_dim + packed_idx * ELTS_PER_THREAD;
#else
    int packed_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int thread_offset = token * token_dim + packed_idx * ELTS_PER_THREAD;
    int clear_token_stride = gridDim.x;
#endif

    cudaGridDependencySynchronize();

// We only use 1 stage for the oneshot allreduce
#ifdef SUPPORT_CGA
    LamportFlags<PackedType, true> flag(buffer_flags, 1);
#else
    LamportFlags<PackedType, false> flag(buffer_flags, 1);
#endif
    T* stage_ptr_mcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcast_ptr, 0));
    T* stage_ptr_local = reinterpret_cast<T*>(flag.getCurLamportBuf(input_ptrs[rank], 0));

    if (packed_idx * ELTS_PER_THREAD >= token_dim)
    {
        flag.clearDirtyLamportBuf(input_ptrs[rank], -1);
        return;
    }

    // ==================== Broadcast tokens to each rank =============================
    PackedVec<PackedType, T> val;
    val.packed = loadPacked<PackedType>(&shard_ptr[thread_offset]);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++)
    {
        if (isNegZero(val.elements[i]))
            val.elements[i] = cuda_cast<T, float>(0.f);
    }

    reinterpret_cast<PackedType*>(&stage_ptr_mcast[token * token_dim * WORLD_SIZE + rank * token_dim])[packed_idx]
        = val.packed;

    flag.cta_arrive();
    // ======================= Lamport Sync and clear the output buffer from previous iteration
    // =============================
    flag.clearDirtyLamportBuf(input_ptrs[rank], -1);

    PackedVec<PackedType, float> values_lamport[WORLD_SIZE];
    while (1)
    {
        bool valid = true;
#pragma unroll
        for (int r = 0; r < WORLD_SIZE; r++)
        {
            values_lamport[r].packed = loadPackedVolatile<PackedType>(
                &stage_ptr_local[token * token_dim * WORLD_SIZE + r * token_dim + packed_idx * ELTS_PER_THREAD]);

#pragma unroll
            for (int i = 0; i < LAMPORT_ELTS_PER_PACKED; i++)
            {
                valid &= !isNegZero(values_lamport[r].elements[i]);
            }
        }
        if (valid)
        {
            break;
        }
    }

    auto values = reinterpret_cast<PackedVec<PackedType, T>*>(values_lamport);
    // ======================= Reduction =============================
    float accum[ELTS_PER_THREAD];
    PackedVec<PackedType, T> packed_accum;
    PackedVec<PackedType, T> residual_in;
    // Move the residual loading up to here
    if constexpr (RESNORM_FUSION)
    {
        residual_in.packed = *reinterpret_cast<PackedType const*>(&residual_in_ptr[thread_offset]);
    }

#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++)
    {
        accum[i] = cuda_cast<float, T>(values[0].elements[i]);
    }

#pragma unroll
    for (int r = 1; r < WORLD_SIZE; r++)
    {
#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i++)
        {
            accum[i] += cuda_cast<float, T>(values[r].elements[i]);
        }
    }

#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++)
    {
        packed_accum.elements[i] = cuda_cast<T, float>(accum[i]);
    }
    cudaTriggerProgrammaticLaunchCompletion();
    if constexpr (RESNORM_FUSION)
    {
        // =============================== Residual ===============================
        packed_accum += residual_in;
        *reinterpret_cast<PackedType*>(&prenormed_ptr[thread_offset]) = packed_accum.packed;
        // =============================== Rmsnorm ================================
        PackedVec<PackedType, T> gamma;
        gamma.packed = *reinterpret_cast<PackedType const*>(&gamma_ptr[packed_idx * ELTS_PER_THREAD]);

        float thread_sum = 0.F;
        __shared__ float shared_val; // Temporary variable to share the sum within block
#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i++)
        {
            // FIXME: Use float square if accuracy issue
            thread_sum += cuda_cast<float, T>(packed_accum.elements[i]) * cuda_cast<float, T>(packed_accum.elements[i]);
        }
        float token_sum = blockReduceSum<float, false>(thread_sum);
#ifdef SUPPORT_CGA
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        if (cluster.num_blocks() > 1)
        {
            // Need to reduce over the entire cluster
            if (threadIdx.x == 0)
            {
                shared_val = token_sum;
                token_sum = 0.f;
            }
            cluster.sync();
            if (threadIdx.x == 0)
            {
                for (int i = 0; i < cluster.num_blocks(); ++i)
                {
                    token_sum += *cluster.map_shared_rank(&shared_val, i);
                }
            }
            cluster.sync();
        }
#endif
        if (threadIdx.x == 0)
        {
            shared_val = rsqrtf(token_sum / token_dim + epsilon);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i++)
        {
            packed_accum.elements[i] = cuda_cast<T, float>(
                cuda_cast<float, T>(packed_accum.elements[i]) * shared_val * cuda_cast<float, T>(gamma.elements[i]));
        }
    }
    reinterpret_cast<PackedType*>(&output_ptr[thread_offset])[0] = packed_accum.packed;
    flag.wait_and_update({static_cast<uint32_t>(num_tokens * token_dim * WORLD_SIZE * ELT_SIZE), 0, 0, 0});
}

using detail::adjust_grid_config;

void oneshot_allreduce_fusion_op(AllReduceFusionParams const& params)
{
    int const num_tokens = params.num_tokens;
    int const token_dim = params.token_dim;
    int const elts_per_thread = sizeof(float4) / getDTypeSize(params.dtype);

#ifdef SUPPORT_CGA
    auto [block_size, cluster_size, loads_per_thread]
        = adjust_grid_config<true>(num_tokens, token_dim, elts_per_thread);
#else
    auto [block_size, cluster_size, loads_per_thread]
        = adjust_grid_config<false>(num_tokens, token_dim, elts_per_thread);
#endif
    dim3 grid(num_tokens, cluster_size, 1);

    TLLM_CHECK_WITH_INFO(block_size <= 1024 && loads_per_thread == 1,
        "Hidden Dimension %d exceeds the maximum supported hidden dimension (%d)", token_dim,
#ifdef SUPPORT_CGA
        1024 * 8 * elts_per_thread);
#else
        1024 * elts_per_thread);
#endif

    TLLM_LOG_DEBUG(
        "[MNNVL AllReduceOneShot] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
        "loads_per_thread: %d, "
        "threads_needed: %d",
        num_tokens, cluster_size, block_size, cluster_size, loads_per_thread, divUp(token_dim, elts_per_thread));

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
#ifdef SUPPORT_CGA
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 1;
    attrs[1].val.clusterDim.y = cluster_size;
    attrs[1].val.clusterDim.z = 1;
#endif

    cudaLaunchConfig_t config{
        .gridDim = grid,
        .blockDim = block_size,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = attrs,
#ifdef SUPPORT_CGA
        .numAttrs = 2,
#else
        .numAttrs = 1,
#endif
    };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, RMSNORM)                                                                \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &oneshot_allreduce_fusion_kernel<WORLD_SIZE, T, RMSNORM>, output,      \
        residual_out, input, residual_in, gamma, uc_ptrs, mc_ptr, num_tokens, token_dim,                               \
        static_cast<float>(params.epsilon), params.rank, params.buffer_flags));
#define DISPATCH_ALLREDUCE_KERNEL(WORLD_SIZE, T)                                                                       \
    if (params.rmsnorm_fusion)                                                                                         \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, true);                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, false);                                                                 \
    }
    // C++17 compatible alternative using a template function
    auto dispatch_impl = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** uc_ptrs = reinterpret_cast<T**>(params.buffer_ptrs_dev);
        T* mc_ptr = reinterpret_cast<T*>(params.multicast_ptr);
        T* output = reinterpret_cast<T*>(params.output);
        T* residual_out = reinterpret_cast<T*>(params.residual_out);
        T const* input = reinterpret_cast<T const*>(params.input);
        T const* residual_in = reinterpret_cast<T const*>(params.residual_in);
        T const* gamma = reinterpret_cast<T const*>(params.gamma);

        switch (params.nranks)
        {
            // FIXME: Do we need other world sizes?
        case 2: DISPATCH_ALLREDUCE_KERNEL(2, T); return true;
        case 4: DISPATCH_ALLREDUCE_KERNEL(4, T); return true;
        case 8: DISPATCH_ALLREDUCE_KERNEL(8, T); return true;
        case 16: DISPATCH_ALLREDUCE_KERNEL(16, T); return true;
        case 32: DISPATCH_ALLREDUCE_KERNEL(32, T); return true;
        case 64: DISPATCH_ALLREDUCE_KERNEL(64, T); return true;
        }
        return false;
    };
#undef LAUNCH_ALLREDUCE_KERNEL
#undef DISPATCH_ALLREDUCE_KERNEL
    bool launched = (params.dtype == nvinfer1::DataType::kBF16 && dispatch_impl((__nv_bfloat16*) nullptr))
        || (params.dtype == nvinfer1::DataType::kFLOAT && dispatch_impl((float*) nullptr))
        || (params.dtype == nvinfer1::DataType::kHALF && dispatch_impl((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "Failed to dispatch MNNVL AllReduceOneShot kernel.");
    }
}

template <uint8_t WORLD_SIZE, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshot_allreduce_kernel(T* output_ptr, T const* shard_ptr, T** input_ptrs,
    T* mcast_ptr, uint32_t const num_tokens, uint32_t const token_dim, uint32_t const rank, uint32_t* buffer_flags,
    bool const wait_for_results)
{
    constexpr int ELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr int LAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
    constexpr uint32_t ELT_SIZE = sizeof(T);

    int packed_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int thread_offset = token * token_dim + packed_idx * ELTS_PER_THREAD;

    int dest_rank = token % WORLD_SIZE;
    int dest_token_offset = token / WORLD_SIZE;

    cudaGridDependencySynchronize();

    LamportFlags<PackedType> flag(buffer_flags, NUM_STAGES);

    T* scatter_buf_local = reinterpret_cast<T*>(flag.getCurLamportBuf(input_ptrs[rank], SCATTER));
    T* scatter_buf_dest = reinterpret_cast<T*>(flag.getCurLamportBuf(input_ptrs[dest_rank], SCATTER));
    T* broadcast_buf_w = reinterpret_cast<T*>(flag.getCurLamportBuf(mcast_ptr, BROADCAST));
    T* broadcast_buf_r = reinterpret_cast<T*>(flag.getCurLamportBuf(input_ptrs[rank], BROADCAST));

    // Make sure the clear function is called before OOB thread exits
    if (packed_idx * ELTS_PER_THREAD >= token_dim)
    {
        flag.clearDirtyLamportBuf(input_ptrs[rank], -1);
        return;
    }

    // =============================== Scatter ===============================

    // Load vectorized data
    PackedVec<PackedType, T> val;
    val.packed = loadPacked<PackedType>(&shard_ptr[thread_offset]);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++)
    {
        if (isNegZero(val.elements[i]))
        {
            val.elements[i] = cuda_cast<T, float>(0.F);
        }
    }

    // Store vectorized data
    reinterpret_cast<PackedType*>(
        &scatter_buf_dest[dest_token_offset * token_dim * WORLD_SIZE + rank * token_dim])[packed_idx]
        = val.packed;

    flag.clearDirtyLamportBuf(input_ptrs[rank], SCATTER);

    // =============================== Reduction and Broadcast ===============================

    if ((token % WORLD_SIZE) == rank)
    {
        int local_token = token / WORLD_SIZE;
        float accum[ELTS_PER_THREAD] = {0.F};

        // Use float as we only check each float value for validity
        PackedVec<PackedType, float> values_lamport[WORLD_SIZE];
        while (1)
        {
            bool valid = true;
#pragma unroll
            for (int r = 0; r < WORLD_SIZE; r++)
            {
                values_lamport[r].packed
                    = loadPackedVolatile<PackedType>(&scatter_buf_local[local_token * token_dim * WORLD_SIZE
                        + r * token_dim + packed_idx * ELTS_PER_THREAD]);

                // Check validity across all elements
#pragma unroll
                for (int i = 0; i < LAMPORT_ELTS_PER_PACKED; i++)
                {
                    valid &= !isNegZero(values_lamport[r].elements[i]);
                }
            }
            if (valid)
            {
                break;
            }
        }

        // Now we view it as the value for reduction
        auto values = reinterpret_cast<PackedVec<PackedType, T>*>(values_lamport);
#pragma unroll
        for (int r = 0; r < WORLD_SIZE; r++)
        {

#pragma unroll
            for (int i = 0; i < ELTS_PER_THREAD; i++)
            {
                accum[i] += cuda_cast<float, T>(values[r].elements[i]);
            }
        }

        // Store vectorized result
        PackedVec<PackedType, T> packed_accum;
#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i++)
        {
            packed_accum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
        reinterpret_cast<PackedType*>(&broadcast_buf_w[token * token_dim])[packed_idx] = packed_accum.packed;
    }
    cudaTriggerProgrammaticLaunchCompletion();
    flag.clearDirtyLamportBuf(input_ptrs[rank], BROADCAST);

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (wait_for_results)
    {
        // Update the atomic counter to indicate the block has read the offsets
        flag.cta_arrive();

        PackedVec<PackedType, float> val_lamport;
        val_lamport.packed = loadPackedVolatile<PackedType>(&broadcast_buf_r[thread_offset]);
        while (isNegZero(val_lamport.elements[0]))
        {
            val_lamport.packed = loadPackedVolatile<PackedType>(&broadcast_buf_r[thread_offset]);
        }
        if (output_ptr)
        {
            reinterpret_cast<PackedType*>(&output_ptr[thread_offset])[0] = val_lamport.packed;
        }

        // Update the buffer flags
        flag.wait_and_update({static_cast<uint32_t>(divUp<uint32_t>(num_tokens, WORLD_SIZE) * WORLD_SIZE * token_dim
                                  * ELT_SIZE),                        // Clear Size for scatter stage
            static_cast<uint32_t>(num_tokens * token_dim * ELT_SIZE), // Clear Size for broadcast stage
            0, 0});
        // If not wait for results, we will rely on the following kernel to update the buffer
    }
}

// This kernel works performant when loads_per_thread is 1.
// For this mode, we are able to support up to 1024 (threads) x 8 (elements) = 8192 hidden dimension.
// There are two options for further scaling up:
//      1. Use CGA if supported. It expands the hidden dimension to 8k x 8 = 64k.
//      2. Set loads_per_thread >1. Which can be used if CGA is not supported. Note that this will be limited by the
//      shared memory size and register count.
template <typename T_IN, typename T_OUT, bool USE_CGA = false, int LOADS_PER_THREAD = 1>
__global__ __launch_bounds__(1024) void rmsnorm_lamport(T_IN* input_plus_residual, T_OUT* output_norm,
    T_IN* buffer_input, T_IN const* gamma, float epsilon, T_IN const* residual, uint32_t num_tokens, uint32_t dim,
    uint32_t world_size, uint32_t* buffer_flags)
{
    // FIXME: Support different types if we'd like to fuse quantization in the future
    static_assert(std::is_same_v<T_IN, T_OUT>, "T_IN and T_OUT must be the same type");
    static int const ELTS_PER_LOAD = sizeof(float4) / sizeof(T_IN);

    uint32_t const token = blockIdx.x;
    uint32_t const block_size = blockDim.x;
    uint32_t const thread_offset = threadIdx.x;

    uint32_t num_threads = block_size;
    uint32_t cluster_size = 1;
    uint32_t block_offset = 0;
    if constexpr (USE_CGA)
    {
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        num_threads = cluster.num_threads();
        cluster_size = cluster.num_blocks();
        block_offset = cluster.block_rank();
    }
    uint32_t const dim_padded = divUp(dim, ELTS_PER_LOAD * num_threads) * ELTS_PER_LOAD * num_threads;
    uint32_t const elems_per_thread = dim_padded / num_threads;
    uint32_t const load_stride = block_size;

    extern __shared__ uint8_t smem[];
    float r_input[LOADS_PER_THREAD * ELTS_PER_LOAD];
    uint32_t offsets[LOADS_PER_THREAD * ELTS_PER_LOAD];

    uint32_t const smem_buffer_size = block_size * elems_per_thread * sizeof(T_IN);
    T_IN* sh_input = (T_IN*) &smem[0];
    T_IN* sh_residual = (T_IN*) &smem[smem_buffer_size];
    T_IN* sh_gamma = (T_IN*) &smem[2 * smem_buffer_size];

    LamportFlags<float4, USE_CGA> flag(buffer_flags, NUM_STAGES);
    T_IN* input = reinterpret_cast<T_IN*>(flag.getCurLamportBuf(reinterpret_cast<void*>(buffer_input), BROADCAST));

    cudaTriggerProgrammaticLaunchCompletion();

    // The offset that current thread should load from. Note that the hidden dimension is split by CGA size and each
    // block loads a contiguous chunk;
    // The size of chunk that each block processes
    uint32_t const block_chunk_size = divUp(dim, cluster_size * ELTS_PER_LOAD) * ELTS_PER_LOAD;
    uint32_t const block_load_offset = token * dim + block_offset * block_chunk_size;

#pragma unroll
    for (uint32_t i = 0; i < LOADS_PER_THREAD; i++)
    {
        // Each block load a contiguous chunk of tokens
        uint32_t const thread_load_offset = (i * load_stride + thread_offset) * ELTS_PER_LOAD;
        offsets[i] = block_load_offset + thread_load_offset;
    }

#pragma unroll
    for (uint32_t i = 0; i < LOADS_PER_THREAD; i++)
    {
        uint32_t const thread_load_offset = (i * load_stride + thread_offset) * ELTS_PER_LOAD;
        if (block_offset * block_chunk_size + thread_load_offset < dim)
        {
            copy_f4(&sh_residual[thread_load_offset], &residual[block_load_offset + thread_load_offset]);
        }
    }
    __pipeline_commit();
#pragma unroll
    for (uint32_t i = 0; i < LOADS_PER_THREAD; i++)
    {
        uint32_t const thread_load_offset = (i * load_stride + thread_offset) * ELTS_PER_LOAD;
        if (block_offset * block_chunk_size + thread_load_offset < dim)
        {
            copy_f4(&sh_gamma[thread_load_offset], &gamma[block_offset * block_chunk_size + thread_load_offset]);
        }
    }
    __pipeline_commit();

    flag.cta_arrive();
    bool valid = false;
    // ACQBLK if not lamport
    while (!valid)
    {
        valid = true;
#pragma unroll
        for (uint32_t i = 0; i < LOADS_PER_THREAD; i++)
        {
            uint32_t thread_load_offset = (i * load_stride + thread_offset) * ELTS_PER_LOAD;

            if (block_offset * block_chunk_size + thread_load_offset < dim)
            {

                float4* dst4 = reinterpret_cast<float4*>(&sh_input[thread_load_offset]);
                float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

                float4 value = loadPackedVolatile<float4>(src4);
                // Assume that the 16B were written atomically, so we only need to check one value
                valid &= !isNegZero(value.x);
                *dst4 = value;
            }
        }
    }

    __pipeline_wait_prior(1);
    __syncthreads();

    float thread_sum = 0.f;
#pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; i++)
    {
        int thread_load_offset = (i * load_stride + thread_offset) * ELTS_PER_LOAD;
        if (block_offset * block_chunk_size + thread_load_offset < dim)
        {
            PackedVec<float4, T_IN> inp{.packed = loadPacked<float4>(&sh_input[thread_load_offset])};
            PackedVec<float4, T_IN> res{.packed = loadPacked<float4>(&sh_residual[thread_load_offset])};

            PackedVec<float4, T_IN> inp_plus_res = inp + res;
#pragma unroll
            for (int j = 0; j < ELTS_PER_LOAD; j++)
            {
                r_input[i * ELTS_PER_LOAD + j] = cuda_cast<float, T_IN>(inp_plus_res.elements[j]);
                thread_sum += cuda_cast<float, T_IN>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
            }

            *reinterpret_cast<float4*>(&input_plus_residual[block_load_offset + thread_load_offset])
                = inp_plus_res.packed;
        }
    }

    __pipeline_wait_prior(0);

    // Sum is only used in thread 0!
    float cluster_sum = blockReduceSum<float, false>(thread_sum);

    float rcp_rms;
    __shared__ float shared_val;
    if constexpr (USE_CGA)
    {
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        if (cluster.num_blocks() > 1)
        {
            // Need to reduce over the entire cluster
            if (threadIdx.x == 0)
            {
                shared_val = cluster_sum;
                cluster_sum = 0.f;
            }
            cluster.sync();
            if (threadIdx.x == 0)
            {
                for (int i = 0; i < cluster_size; ++i)
                {
                    cluster_sum += *cluster.map_shared_rank(&shared_val, i);
                }
            }
            cluster.sync();
        }
    }

    if (threadIdx.x == 0)
    {
        shared_val = rsqrtf(cluster_sum / dim + epsilon);
    }
    __syncthreads();
    rcp_rms = shared_val;

#pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; i++)
    {
        PackedVec<float4, T_OUT> r_out;
        uint32_t thread_load_offset = (i * load_stride + thread_offset) * ELTS_PER_LOAD;
        if (block_offset * block_chunk_size + thread_load_offset < dim)
        {
            PackedVec<float4, T_IN> gamma = {.packed = loadPacked<float4>(&sh_gamma[thread_load_offset])};

#pragma unroll
            for (uint32_t j = 0; j < ELTS_PER_LOAD; j++)
            {
                r_out.elements[j]
                    = cuda_cast<T_OUT, float>(cuda_cast<float, T_IN>(gamma.elements[j]) * r_input[j] * rcp_rms);
            }

            *reinterpret_cast<float4*>(&output_norm[block_load_offset + thread_load_offset]) = r_out.packed;
        }
    }
    constexpr int ELTS_SIZE = sizeof(T_IN);

    // Update the buffer pointers
    // FIXME: We round num_tokens to 64 to avoid passing in the world size into this kernel; Check this when we need
    // world size >64
    flag.wait_and_update({static_cast<uint32_t>(divUp<uint32_t>(num_tokens, world_size) * world_size * dim * ELTS_SIZE),
        static_cast<uint32_t>(num_tokens * dim * ELTS_SIZE), 0, 0});
}

void twoshot_allreduce_fusion_op(AllReduceFusionParams const& params)
{
    int const num_tokens = params.num_tokens;
    int const token_dim = params.token_dim;
    int const num_elts_per_thread = sizeof(float4) / getDTypeSize(params.dtype);
    TLLM_CHECK_WITH_INFO(token_dim % num_elts_per_thread == 0,
        "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d", num_elts_per_thread);

    int const ar_num_threads = divUp(token_dim, num_elts_per_thread);
    int const ar_num_blocks_per_token = divUp(ar_num_threads, 128);

    dim3 ar_grid(num_tokens, ar_num_blocks_per_token);

    cudaLaunchAttribute ar_attrs[1];
    ar_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    ar_attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;

    cudaLaunchConfig_t ar_config{
        .gridDim = ar_grid,
        .blockDim = 128,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = ar_attrs,
        .numAttrs = 1,
    };

    TLLM_LOG_DEBUG("[MNNVL AllReduceTwoShot] Dispatch: grid size: (%d, %d, 1), block_size: 128", num_tokens,
        ar_num_blocks_per_token);

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T)                                                                         \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&ar_config, &twoshot_allreduce_kernel<WORLD_SIZE, T>, output, input, uc_ptrs,   \
        mcast_ptr, num_tokens, token_dim, params.rank, params.buffer_flags, (!params.rmsnorm_fusion)));
    auto dispatch_ar = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** uc_ptrs = reinterpret_cast<T**>(params.buffer_ptrs_dev);
        T* mcast_ptr = reinterpret_cast<T*>(params.multicast_ptr);
        T* output = reinterpret_cast<T*>(params.output);
        T const* input = reinterpret_cast<T const*>(params.input);
        switch (params.nranks)
        {
        case 2: LAUNCH_ALLREDUCE_KERNEL(2, T); return true;
        case 4: LAUNCH_ALLREDUCE_KERNEL(4, T); return true;
        case 8: LAUNCH_ALLREDUCE_KERNEL(8, T); return true;
        case 16: LAUNCH_ALLREDUCE_KERNEL(16, T); return true;
        case 32: LAUNCH_ALLREDUCE_KERNEL(32, T); return true;
        case 64: LAUNCH_ALLREDUCE_KERNEL(64, T); return true;
        }
        return false;
    };

#undef LAUNCH_ALLREDUCE_KERNEL

    bool launched = (params.dtype == nvinfer1::DataType::kFLOAT && dispatch_ar((float*) nullptr))
        || (params.dtype == nvinfer1::DataType::kBF16 && dispatch_ar((__nv_bfloat16*) nullptr))
        || (params.dtype == nvinfer1::DataType::kHALF && dispatch_ar((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch twoshot_allreduce kernel.");
    }
    // Launch the rmsnorm lamport kernel if fusion is enabled
    if (params.rmsnorm_fusion)
    {
#ifdef SUPPORT_CGA
        auto grid_config = adjust_grid_config<true>(num_tokens, token_dim, num_elts_per_thread);
#else
        auto grid_config = adjust_grid_config<false>(num_tokens, token_dim, num_elts_per_thread);
#endif
        int rn_block_size = std::get<0>(grid_config);
        int rn_cluster_size = std::get<1>(grid_config);
        int rn_loads_per_thread = std::get<2>(grid_config);

        int rn_num_threads = rn_cluster_size * rn_block_size;
        dim3 rn_grid(num_tokens, rn_cluster_size, 1);
        cudaLaunchConfig_t rn_config;
        cudaLaunchAttribute rn_attrs[2];
        rn_config.stream = params.stream;
        rn_config.gridDim = rn_grid;
        rn_config.blockDim = rn_block_size;
        rn_config.attrs = rn_attrs;
        rn_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        rn_attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
#ifndef DISABLE_CGA
        rn_attrs[1].id = cudaLaunchAttributeClusterDimension;
        rn_attrs[1].val.clusterDim.x = 1;
        rn_attrs[1].val.clusterDim.y = rn_cluster_size;
        rn_attrs[1].val.clusterDim.z = 1;
        rn_config.numAttrs = 2;
#else
        rn_config.numAttrs = 1;
#endif

        bool const rn_use_cga = rn_cluster_size > 1;
        int const dim_padded
            = divUp(token_dim, num_elts_per_thread * rn_num_threads) * num_elts_per_thread * rn_num_threads;
        int const iters = dim_padded / rn_num_threads;
        assert(rn_loads_per_thread == iters / num_elts_per_thread);
        size_t const shmem_size = 3 * rn_block_size * iters * getDTypeSize(params.dtype);

        TLLM_LOG_DEBUG(
            "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
            "loads_per_thread: %d, "
            "threads_needed: %d",
            num_tokens, rn_cluster_size, rn_block_size, rn_cluster_size, rn_loads_per_thread,
            divUp(token_dim, num_elts_per_thread));

#define RUN_RMSNORM_KERNEL(T_IN, T_OUT, USE_CGA, LOADS_PER_THREAD)                                                     \
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(&rmsnorm_lamport<T_IN, T_OUT, USE_CGA, LOADS_PER_THREAD>,                     \
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));                                                     \
    rn_config.dynamicSmemBytes = shmem_size;                                                                           \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&rn_config, &rmsnorm_lamport<T_IN, T_OUT, USE_CGA, LOADS_PER_THREAD>,           \
        residual_out, output, buffer_input, gamma, static_cast<float>(params.epsilon), residual_in, num_tokens,        \
        token_dim, params.nranks, params.buffer_flags));

        // C++ 17 does not support capturing structured bindings
        auto dispatch_rn = [&, rn_loads_per_thread](auto* type_ptr)
        {
            using T_IN = std::remove_pointer_t<decltype(type_ptr)>;
            using T_OUT = T_IN;
            T_OUT* residual_out = reinterpret_cast<T_OUT*>(params.residual_out);
            T_OUT* output = reinterpret_cast<T_OUT*>(params.output);
            T_IN* buffer_input = reinterpret_cast<T_IN*>(params.buffer_ptr_local);
            T_IN const* gamma = reinterpret_cast<T_IN const*>(params.gamma);
            T_IN const* residual_in = reinterpret_cast<T_IN const*>(params.residual_in);
            if (rn_use_cga)
            {
                RUN_RMSNORM_KERNEL(T_IN, T_OUT, true, 1);
            }
            else
            {
                switch (rn_loads_per_thread)
                {
                case 1: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 1); break;
                case 2: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 2); break;
                case 3: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 3); break;
                case 4: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 4); break;
                case 5: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 5); break;
                case 6: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 6); break;
                case 7: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 7); break;
                case 8: RUN_RMSNORM_KERNEL(T_IN, T_OUT, false, 8); break;
                default: return false;
                }
            }
            return true;
        };

        launched = (params.dtype == nvinfer1::DataType::kFLOAT && dispatch_rn((float*) nullptr))
            || (params.dtype == nvinfer1::DataType::kBF16 && dispatch_rn((__nv_bfloat16*) nullptr))
            || (params.dtype == nvinfer1::DataType::kHALF && dispatch_rn((__nv_half*) nullptr));
        if (!launched)
        {
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch rmsnorm lamport kernel.");
        }
#undef RUN_RMSNORM_KERNEL
    }
}

// Avoid polluting the global namespace
#ifdef SUPPORT_CGA
#undef SUPPORT_CGA
#endif

} // namespace tensorrt_llm::kernels::mnnvl
