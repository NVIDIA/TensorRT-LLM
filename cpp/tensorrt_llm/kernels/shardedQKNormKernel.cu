/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

/*
 * The SyncComm and Barrier structs below are adapted from:
 *   tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. Apache-2.0 License.
 */

#include "tensorrt_llm/kernels/shardedQKNormKernel.h"

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace tensorrt_llm::kernels
{

// ---------------------------------------------------------------------------
// Barrier flag count — must match kBarrierFlagCount in allReduceFusionKernels.h
// ---------------------------------------------------------------------------
static constexpr int kBarrierFlagCount = 256;

// ---------------------------------------------------------------------------
// SyncComm: manages per-block counter and flag, and pointers to peer buffers.
// Adapted from allReduceFusionKernels.cu (tensorrt_llm::kernels::ar_fusion).
// ---------------------------------------------------------------------------
template <int NRanks>
struct SyncComm
{
    __device__ __forceinline__ SyncComm(void** workspace)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r)
        {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x)
            {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    void* comm_bufs[NRanks];
    void* barrier_flags[NRanks];
    int flag_value;
};

// ---------------------------------------------------------------------------
// Barrier: per-rank cross-GPU synchronization using NVLink store/load.
// Adapted from allReduceFusionKernels.cu (tensorrt_llm::kernels::ar_fusion).
// ---------------------------------------------------------------------------
template <int NRanks>
class Barrier
{
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm)
    {
        if (threadIdx.x < NRanks)
        {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
            m_current_flag
                = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync()
    {
        __syncthreads();
        if (threadIdx.x < NRanks)
        {
            m_flag_value = next_flag(m_flag_value);
            // Signal all blocks up to kBarrierFlagCount to avoid ABA problems.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount; flag_idx += gridDim.x)
            {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value))
            {
            }
        }
        __syncthreads();
    }

protected:
    __device__ __forceinline__ void st_flag(int* addr, int flag)
    {
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
    }

    __device__ __forceinline__ int ld_flag(int* addr)
    {
        int flag;
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
        return flag;
    }

    __device__ __forceinline__ int next_flag(int flag)
    {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag)
    {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    int m_flag_value;

private:
    int* m_target_flag;
    int* m_current_flag;
};

// ---------------------------------------------------------------------------
// Warp-level sum reduction (used in block reduction below).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level sum reduction for two float values simultaneously.
// Returns the block-wide sum in thread 0; all threads should call this.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void blockReduceSum2(float& val0, float& val1)
{
    __shared__ float shared0[33];
    __shared__ float shared1[33];

    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val0 = warpReduceSum(val0);
    val1 = warpReduceSum(val1);

    if (lane == 0)
    {
        shared0[wid] = val0;
        shared1[wid] = val1;
    }
    __syncthreads();

    int nwarps = (blockDim.x + 31) / 32;
    val0 = (threadIdx.x < nwarps) ? shared0[lane] : 0.f;
    val1 = (threadIdx.x < nwarps) ? shared1[lane] : 0.f;

    val0 = warpReduceSum(val0);
    val1 = warpReduceSum(val1);
}

// ---------------------------------------------------------------------------
// Main fused kernel: one block per token.
// ---------------------------------------------------------------------------
template <int NRanks, typename DType>
__global__ void shardedQKNormKernel(DType const* __restrict__ q_in, DType const* __restrict__ k_in,
    DType* __restrict__ q_out, DType* __restrict__ k_out, float const* __restrict__ weight_q,
    float const* __restrict__ weight_k, void** workspace, int n_tokens, int local_q_dim, int local_k_dim, int q_stride,
    int k_stride, int world_size, int rank, float eps)
{
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    // -----------------------------------------------------------------------
    // Phase 1: compute local sum-of-squares for this token's Q and K rows.
    // Input uses q_stride / k_stride (>= local dim) to allow a view into a
    // fused [N, total_dim] tensor (e.g. Q+K+V output of a fused GEMM).
    // -----------------------------------------------------------------------
    float q_sumsq = 0.f;
    float k_sumsq = 0.f;

    for (int i = threadIdx.x; i < local_q_dim; i += blockDim.x)
    {
        float v = static_cast<float>(q_in[token_idx * q_stride + i]);
        q_sumsq += v * v;
    }
    for (int i = threadIdx.x; i < local_k_dim; i += blockDim.x)
    {
        float v = static_cast<float>(k_in[token_idx * k_stride + i]);
        k_sumsq += v * v;
    }

    blockReduceSum2(q_sumsq, k_sumsq);

    // Thread 0 writes the packed [q_sumsq, k_sumsq] into this rank's comm buffer.
    if (threadIdx.x == 0)
    {
        float2* comm_slot = reinterpret_cast<float2*>(workspace[rank]) + token_idx;
        *comm_slot = make_float2(q_sumsq, k_sumsq);
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Phase 2: cross-GPU barrier — wait for all NRanks to write their variances.
    // -----------------------------------------------------------------------
    SyncComm<NRanks> comm(workspace);
    Barrier<NRanks> barrier(rank, comm);
    barrier.sync();
    comm.update(comm.flag_value + 1);
    // After update, all blocks on all ranks have arrived.

    // -----------------------------------------------------------------------
    // Phase 3: each rank independently reads all peers' packed variances and
    //          sums them to get global_q_sumsq, global_k_sumsq.
    // -----------------------------------------------------------------------
    __shared__ float2 s_global_stats;
    if (threadIdx.x == 0)
    {
        float global_q = 0.f, global_k = 0.f;
        for (int r = 0; r < NRanks; ++r)
        {
            float2 peer = reinterpret_cast<float2 const*>(workspace[r])[token_idx];
            global_q += peer.x;
            global_k += peer.y;
        }
        s_global_stats = make_float2(global_q, global_k);
    }
    __syncthreads();

    float global_q_sumsq = s_global_stats.x;
    float global_k_sumsq = s_global_stats.y;

    // -----------------------------------------------------------------------
    // Phase 4: normalize Q and K using global stats.
    // -----------------------------------------------------------------------
    float global_q_count = static_cast<float>(local_q_dim * world_size);
    float global_k_count = static_cast<float>(local_k_dim * world_size);
    float rstd_q = rsqrtf(global_q_sumsq / global_q_count + eps);
    float rstd_k = rsqrtf(global_k_sumsq / global_k_count + eps);

    // Output rows are packed (stride = local_dim); input rows may be strided.
    for (int i = threadIdx.x; i < local_q_dim; i += blockDim.x)
    {
        float v = static_cast<float>(q_in[token_idx * q_stride + i]);
        q_out[token_idx * local_q_dim + i] = static_cast<DType>(v * rstd_q * weight_q[i]);
    }
    for (int i = threadIdx.x; i < local_k_dim; i += blockDim.x)
    {
        float v = static_cast<float>(k_in[token_idx * k_stride + i]);
        k_out[token_idx * local_k_dim + i] = static_cast<DType>(v * rstd_k * weight_k[i]);
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <int NRanks, typename DType>
static void launchTyped(void* q_in, void* k_in, void* q_out, void* k_out, void* weight_q, void* weight_k,
    void** workspace, int n_tokens, int local_q_dim, int local_k_dim, int q_stride, int k_stride, int world_size,
    int rank, float eps, cudaStream_t stream)
{
    // Block size: cover max(local_q_dim, local_k_dim), rounded to warp, capped at 512.
    int max_dim = (local_q_dim > local_k_dim) ? local_q_dim : local_k_dim;
    int block_size = ((max_dim + 31) / 32) * 32;
    if (block_size > 512)
        block_size = 512;
    if (block_size < 32)
        block_size = 32;

    // Grid: one block per token.
    shardedQKNormKernel<NRanks, DType><<<n_tokens, block_size, 0, stream>>>(reinterpret_cast<DType const*>(q_in),
        reinterpret_cast<DType const*>(k_in), reinterpret_cast<DType*>(q_out), reinterpret_cast<DType*>(k_out),
        reinterpret_cast<float const*>(weight_q), reinterpret_cast<float const*>(weight_k), workspace, n_tokens,
        local_q_dim, local_k_dim, q_stride, k_stride, world_size, rank, eps);
}

void launchShardedQKNormKernel(void* q_in, void* k_in, void* q_out, void* k_out, void* weight_q, void* weight_k,
    void** workspace, int n_tokens, int local_q_dim, int local_k_dim, int q_stride, int k_stride, int world_size,
    int rank, float eps, bool is_bf16, cudaStream_t stream)
{
    if (n_tokens > kBarrierFlagCount)
    {
        throw std::runtime_error("shardedQKNormKernel: n_tokens (" + std::to_string(n_tokens)
            + ") exceeds kBarrierFlagCount (" + std::to_string(kBarrierFlagCount)
            + "). Use NCCL-based sharded_rmsnorm for large batch sizes.");
    }

#define DISPATCH_RANKS_DTYPE(NRANKS, DTYPE)                                                                            \
    launchTyped<NRANKS, DTYPE>(q_in, k_in, q_out, k_out, weight_q, weight_k, workspace, n_tokens, local_q_dim,         \
        local_k_dim, q_stride, k_stride, world_size, rank, eps, stream)

    if (is_bf16)
    {
        switch (world_size)
        {
        case 2: DISPATCH_RANKS_DTYPE(2, __nv_bfloat16); break;
        case 4: DISPATCH_RANKS_DTYPE(4, __nv_bfloat16); break;
        case 8: DISPATCH_RANKS_DTYPE(8, __nv_bfloat16); break;
        default:
            throw std::runtime_error(
                "shardedQKNormKernel: unsupported world_size=" + std::to_string(world_size) + " for bf16");
        }
    }
    else
    {
        switch (world_size)
        {
        case 2: DISPATCH_RANKS_DTYPE(2, half); break;
        case 4: DISPATCH_RANKS_DTYPE(4, half); break;
        case 8: DISPATCH_RANKS_DTYPE(8, half); break;
        default:
            throw std::runtime_error(
                "shardedQKNormKernel: unsupported world_size=" + std::to_string(world_size) + " for fp16");
        }
    }

#undef DISPATCH_RANKS_DTYPE
}

} // namespace tensorrt_llm::kernels
