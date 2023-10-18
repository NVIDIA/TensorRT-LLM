/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include <tuple>

namespace tensorrt_llm::kernels
{

using tensorrt_llm::common::hadd2;
using tensorrt_llm::common::datatype_enum;
using tensorrt_llm::common::divUp;

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t myHadd2(const uint32_t& a, const uint32_t& b)
{
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t fadd(const uint32_t& a, const uint32_t& b)
{
    uint32_t c;
    asm volatile("add.f32 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void st_flag_release(uint32_t& flag, uint32_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void ld_flag_acquire(uint32_t& flag, uint32_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.global.volatile.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Type Converter that packs data format to 128 bits data type
template <typename T>
struct ARTypeConverter
{
    using Type = uint4;
};

#ifdef ENABLE_BF16
template <>
struct ARTypeConverter<__nv_bfloat16>
{
    using Type = bf168;
};
#endif

// add two 128b data
template <typename T_IN, typename T_COMP>
inline __device__ T_IN add128b(T_IN a, T_IN b);

template <>
inline __device__ uint4 add128b<uint4, uint16_t>(uint4 a, uint4 b)
{
    uint4 c;
    c.x = myHadd2(a.x, b.x);
    c.y = myHadd2(a.y, b.y);
    c.z = myHadd2(a.z, b.z);
    c.w = myHadd2(a.w, b.w);
    return c;
}

template <>
inline __device__ uint4 add128b<uint4, uint32_t>(uint4 a, uint4 b)
{
    uint4 c;
    c.x = fadd(a.x, b.x);
    c.y = fadd(a.y, b.y);
    c.z = fadd(a.z, b.z);
    c.w = fadd(a.w, b.w);
    return c;
}

#ifdef ENABLE_BF16
template <>
inline __device__ bf168 add128b<bf168, __nv_bfloat16>(bf168 a, bf168 b)
{
    bf168 c;
    c.x = hadd2(a.x, b.x);
    c.y = hadd2(a.y, b.y);
    c.z = hadd2(a.z, b.z);
    c.w = hadd2(a.w, b.w);
    return c;
}
#endif

// init 128bits data with 0
template <typename T>
inline __device__ T init_packed_type();

template <>
inline __device__ uint4 init_packed_type()
{
    return make_uint4(0u, 0u, 0u, 0u);
}

#ifdef ENABLE_BF16
template <>
inline __device__ bf168 init_packed_type()
{
    bf168 val;
    uint4& val_u = reinterpret_cast<uint4&>(val);
    val_u = make_uint4(0u, 0u, 0u, 0u);
    return val;
}
#endif

__inline__ __device__ void multi_gpu_barrier(
    uint32_t** signals, const uint32_t flag, const size_t rank, const size_t world_size, const int tidx, const int bidx)
{
    // At the end of the function, we now that has least block 0 from all others GPUs have reached that point.
    volatile uint32_t* my_signals = signals[rank];
    if (tidx < world_size)
    {
        // The 1st block notifies the other ranks.
        if (bidx == 0)
        {
            signals[tidx][rank] = flag;
        }

        // Busy-wait until all ranks are ready.
        while (my_signals[tidx] != flag)
        {
        }
    }

    // Make sure we can move on...
    __syncthreads();
}

__global__ void multiGpuBarrierKernel(AllReduceParams params)
{
    multi_gpu_barrier(params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, params.ranks_per_node,
        threadIdx.x, blockIdx.x);
}

template <typename T, int RANKS_PER_NODE>
static __global__ void oneShotAllReduceKernel(AllReduceParams params)
{
    const int bidx = blockIdx.x;
    const int tidx = threadIdx.x;

    // The number of elements packed into one for comms
    static constexpr int NUM_ELTS = std::is_same<T, uint32_t>::value ? 4 : 8;

    // Packed data type for comms
    using PackedType = typename ARTypeConverter<T>::Type;

    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

    // The source pointers. Distributed round-robin for the different warps.
    const T* src_d[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
    {
        int rank = (params.local_rank + ii) % RANKS_PER_NODE;
        src_d[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
    }

    // The location in the destination array (load 8 fp16 or load 4 fp32 using LDG.128).
    size_t offset = bidx * params.elts_per_block + tidx * NUM_ELTS;
    // The end of the segment computed by that block.
    size_t max_offset = std::min((bidx + 1) * params.elts_per_block, params.elts_per_rank);

    // Each block accumulates the values from the different GPUs on the same node.
    for (size_t iter_offset = offset; iter_offset < max_offset; iter_offset += blockDim.x * NUM_ELTS)
    {
        // Iterate over the different ranks/devices on the node to load the values.
        PackedType vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            vals[ii] = reinterpret_cast<const PackedType*>(&src_d[ii][iter_offset])[0];
        }

        // Sum the values from the different ranks.
        PackedType sums = init_packed_type<PackedType>();
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            sums = add128b<PackedType, T>(sums, vals[ii]);
        }

        // Store to the destination buffer.
        reinterpret_cast<PackedType*>(&reinterpret_cast<T*>(params.local_output_buffer_ptr)[iter_offset])[0] = sums;
    }
}

template <typename T, int RANKS_PER_NODE>
static __global__ void twoShotAllReduceKernel(AllReduceParams params)
{

    // The block index.
    const int bidx = blockIdx.x;
    // The thread index with the block.
    const int tidx = threadIdx.x;

    // The number of elements packed into one for comms
    static constexpr int NUM_ELTS = std::is_same<T, uint32_t>::value ? 4 : 8;

    // Packed data type for comms
    using PackedType = typename ARTypeConverter<T>::Type;

    // The location in the destination array (load 8 fp16 or load 4 fp32 using LDG.128).
    const size_t block_offset = bidx * params.elts_per_block + tidx * NUM_ELTS;
    const size_t block_start = params.rank_offset + block_offset;
    // The end of the segment computed by that block.
    size_t max_offset = min(block_start + params.elts_per_block, params.rank_offset + params.elts_per_rank);

    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

    // The source pointers. Distributed round-robin for the different warps.
    T* src_d[RANKS_PER_NODE];
    // The destination ranks for round-robin gathering
    size_t dst_rank[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
    {
        int rank = (params.local_rank + ii) % RANKS_PER_NODE;
        src_d[ii] = reinterpret_cast<T*>(params.peer_comm_buffer_ptrs[rank]);
        dst_rank[ii] = rank;
    }

    // Each block accumulates the values from the different GPUs on the same node.
    for (size_t local_offset = block_start; local_offset < max_offset; local_offset += blockDim.x * NUM_ELTS)
    {

        // Iterate over the different ranks/devices on the node to load the values.
        PackedType vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            vals[ii] = reinterpret_cast<const PackedType*>(&src_d[ii][local_offset])[0];
        }

        // Sum the values from the different ranks.
        PackedType sums = init_packed_type<PackedType>();
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            sums = add128b<PackedType, T>(sums, vals[ii]);
        }

        // Store to the local buffer.
        reinterpret_cast<PackedType*>(&src_d[0][local_offset])[0] = sums;
    }

    // sync threads to make sure all block threads have the sums
    __syncthreads();

    // barriers among the blocks with the same idx (release-acquire semantics)
    if (tidx < RANKS_PER_NODE)
    {
        // The all blocks notifies the other ranks.
        uint32_t flag_block_offset = RANKS_PER_NODE + bidx * RANKS_PER_NODE;
        st_flag_release(params.barrier_flag, params.peer_barrier_ptrs_in[tidx] + flag_block_offset + params.local_rank);

        // Busy-wait until all ranks are ready.
        uint32_t rank_barrier = 0;
        uint32_t* peer_barrier_d = params.peer_barrier_ptrs_in[params.local_rank] + flag_block_offset + tidx;
        do
        {
            ld_flag_acquire(rank_barrier, peer_barrier_d);
        } while (rank_barrier != params.barrier_flag);
    }

    // sync threads to make sure all other ranks has the final partial results
    __syncthreads();

    size_t max_block_offset = min(block_offset + params.elts_per_block, params.elts_per_rank);
    // Gather all needed elts from other intra-node ranks
    for (size_t local_offset = block_offset; local_offset < max_block_offset; local_offset += blockDim.x * NUM_ELTS)
    {
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            // use round-robin gathering from other ranks
            size_t offset_rank = dst_rank[ii] * params.elts_per_rank + local_offset;
            if (offset_rank >= params.elts_total)
            {
                continue;
            }
            reinterpret_cast<PackedType*>(&reinterpret_cast<T*>(params.local_output_buffer_ptr)[offset_rank])[0]
                = reinterpret_cast<PackedType*>(&src_d[ii][offset_rank])[0];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<int, int> kernelLaunchConfig(AllReduceStrategyType algo, AllReduceParams& param, size_t elts_per_thread)
{
    TLLM_CHECK(param.elts_total % elts_per_thread == 0);

    int blocks_per_grid = 1, threads_per_block = DEFAULT_BLOCK_SIZE;

    const size_t total_threads = param.elts_total / elts_per_thread;
    switch (algo)
    {
    case AllReduceStrategyType::ONESHOT:
    {     // one stage all reduce algo
        if (total_threads <= DEFAULT_BLOCK_SIZE)
        { // local reduce
            threads_per_block = WARP_SIZE * divUp(total_threads, WARP_SIZE);
            blocks_per_grid = 1;
        }
        else
        { // local reduce
            threads_per_block = DEFAULT_BLOCK_SIZE;
            blocks_per_grid = divUp(total_threads, DEFAULT_BLOCK_SIZE);
            blocks_per_grid = std::min(static_cast<int>(MAX_ALL_REDUCE_BLOCKS), blocks_per_grid);
        }
        param.elts_per_rank = param.elts_total;
        param.elts_per_block = elts_per_thread * divUp(param.elts_per_rank, elts_per_thread * blocks_per_grid);
        break;
    }
    case AllReduceStrategyType::TWOSHOT:
    { // two stage all reduce algo
        const size_t elts_per_rank = param.elts_total / param.ranks_per_node;
        TLLM_CHECK(elts_per_rank % elts_per_thread == 0);

        size_t total_threads = elts_per_rank / elts_per_thread;
        total_threads = WARP_SIZE * ((total_threads + WARP_SIZE - 1) / WARP_SIZE);
        TLLM_CHECK(total_threads % WARP_SIZE == 0);

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
        param.elts_per_rank = param.elts_total / param.ranks_per_node;
        param.elts_per_block = param.elts_per_rank / blocks_per_grid;
        param.elts_per_block = elts_per_thread * divUp(param.elts_per_block, elts_per_thread);
        param.rank_offset = param.rank * param.elts_per_rank;
        break;
    }
    default: TLLM_THROW("Algorithm not supported here.");
    }

    return std::make_tuple(blocks_per_grid, threads_per_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int RANKS_PER_NODE>
void dispatchARKernels(
    AllReduceStrategyType algo, AllReduceParams& param, int blocks_per_grid, int threads_per_block, cudaStream_t stream)
{
    if (algo == AllReduceStrategyType::ONESHOT)
    {
        oneShotAllReduceKernel<T, RANKS_PER_NODE><<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
    }
    else
    {
        twoShotAllReduceKernel<T, RANKS_PER_NODE><<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
    }
}

template <typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams& param, AllReduceStrategyType strat, cudaStream_t stream)
{
    TLLM_CHECK(strat == AllReduceStrategyType::ONESHOT || strat == AllReduceStrategyType::TWOSHOT);
    sync_check_cuda_error();

    size_t elts_per_thread = 16 / sizeof(T);
    auto [blocks_per_grid, threads_per_block] = kernelLaunchConfig(strat, param, elts_per_thread);
    switch (param.ranks_per_node)
    {
    case 2: dispatchARKernels<T, 2>(strat, param, blocks_per_grid, threads_per_block, stream); break;
    case 4: dispatchARKernels<T, 4>(strat, param, blocks_per_grid, threads_per_block, stream); break;
    case 6: dispatchARKernels<T, 6>(strat, param, blocks_per_grid, threads_per_block, stream); break;
    case 8: dispatchARKernels<T, 8>(strat, param, blocks_per_grid, threads_per_block, stream); break;
    default: break;
    }
    sync_check_cuda_error();
}

void invokeMultiGpuBarrier(AllReduceParams& param, cudaStream_t stream)
{
    multiGpuBarrierKernel<<<1, param.ranks_per_node, 0, stream>>>(param);
}

AllReduceParams AllReduceParams::deserialize(const int32_t* buffer, size_t tpSize, size_t tpRank, uint32_t flag_value)
{
    void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);
    AllReduceParams params;

    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_comm_buffer_ptrs[i] = buffer_ptrs[i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[tpSize + i]);
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[2 * tpSize + i]);
    }
    params.barrier_flag = flag_value;
    params.ranks_per_node = tpSize;
    params.rank = tpRank;
    params.local_rank = tpRank;

    return params;
}

void customAllReduce(kernels::AllReduceParams& params, void* data, size_t elts, size_t size_per_elem,
    datatype_enum dataType, AllReduceStrategyType strat, cudaStream_t stream)
{
    params.local_output_buffer_ptr = data;
    params.elts_total = elts;

    if (dataType == datatype_enum::TYPE_FP32)
    {
        using T = CustomARCommTypeConverter<float>::Type;
        kernels::invokeOneOrTwoShotAllReduceKernel<T>(params, strat, stream);
    }
    else if (dataType == datatype_enum::TYPE_FP16)
    {
        using T = CustomARCommTypeConverter<half>::Type;
        kernels::invokeOneOrTwoShotAllReduceKernel<T>(params, strat, stream);
    }
    else if (dataType == datatype_enum::TYPE_BF16)
    {
        using T = CustomARCommTypeConverter<__nv_bfloat16>::Type;
        kernels::invokeOneOrTwoShotAllReduceKernel<T>(params, strat, stream);
    }
    else
    {
        TLLM_THROW("Unsupported dataType for customAllReduce");
    }
}

} // namespace tensorrt_llm::kernels
