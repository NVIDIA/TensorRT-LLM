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
#include <tuple>

namespace tensorrt_llm::kernels
{

using tensorrt_llm::common::datatype_enum;
using tensorrt_llm::common::divUp;

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
    static constexpr int NUM_ELTS = 16 / sizeof(T);

    // Packed data type for comms
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

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
        PackedStruct vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            vals[ii].packed = *reinterpret_cast<const int4*>(&src_d[ii][iter_offset]);
        }

        // Sum the values from the different ranks.
        PackedStruct sums;
        sums.packed = {0, 0, 0, 0};
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            sums.packed = add128b(sums, vals[ii]);
        }

        // Store to the destination buffer.
        *reinterpret_cast<int4*>(&reinterpret_cast<T*>(params.local_output_buffer_ptr)[iter_offset]) = sums.packed;
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
    static constexpr int NUM_ELTS = 16 / sizeof(T);

    // Packed data type for comms
    using PackedType = typename PackedOn16Bytes<T>::Type;

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
            vals[ii].packed = *reinterpret_cast<const int4*>(&src_d[ii][local_offset]);
        }

        // Sum the values from the different ranks.
        PackedType sums;
        sums.packed = {0, 0, 0, 0};
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            sums.packed = add128b(sums, vals[ii]);
        }

        // Store to the local buffer.
        *reinterpret_cast<int4*>(&src_d[0][local_offset]) = sums.packed;
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
            *reinterpret_cast<int4*>(&reinterpret_cast<T*>(params.local_output_buffer_ptr)[offset_rank])
                = *reinterpret_cast<int4*>(&src_d[ii][offset_rank]);
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
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    const auto buffer_offset = (flag_value % 2 == 0) ? 0 : tpSize;

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
        kernels::invokeOneOrTwoShotAllReduceKernel<float>(params, strat, stream);
    }
    else if (dataType == datatype_enum::TYPE_FP16)
    {
        kernels::invokeOneOrTwoShotAllReduceKernel<half>(params, strat, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == datatype_enum::TYPE_BF16)
    {
        kernels::invokeOneOrTwoShotAllReduceKernel<__nv_bfloat16>(params, strat, stream);
    }
#endif
    else
    {
        TLLM_THROW("Unsupported dataType for customAllReduce");
    }
}

} // namespace tensorrt_llm::kernels
