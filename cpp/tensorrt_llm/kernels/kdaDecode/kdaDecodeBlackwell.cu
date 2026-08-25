/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace
{

namespace cg = cooperative_groups;

constexpr int kHeads = 12;
constexpr int kDim = 128;
constexpr int kFlat = kHeads * kDim;
constexpr int kKernelWidth = 4;
constexpr int kConvCacheWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkRows = 32;
constexpr int kChunks = kDim / kChunkRows;
constexpr float kLowerBound = -5.0f;
constexpr float kQScale = 0.08838834764831845f;
constexpr float kOutputNormEpsilon = 1.0e-5f;

enum class KernelSchedule : int
{
    kSingleCtaCpAsync = 0,
    kSingleCtaTwoStageCpAsyncBulk = 1,
    kSingleCtaFourStageCpAsyncBulk = 2,
    kFourCtaThreadBlockClusterCpAsync = 3,
};

template <bool kUseCpAsyncBulk, int kStages>
struct StateSharedStorage
{
    alignas(16) float data[kStages][kChunkRows][kDim];
};

template <int kStages>
struct StateSharedStorage<true, kStages>
{
};

template <bool kUseCpAsyncBulk>
struct CpAsyncBulkBarrierStorage
{
};

template <>
struct CpAsyncBulkBarrierStorage<true>
{
    alignas(8) uint64_t ready[kChunks];
};

struct Sum2
{
    float x;
    float y;
};

struct KernelArguments
{
    __nv_bfloat16 const* x_q;
    __nv_bfloat16 const* x_k;
    __nv_bfloat16 const* x_v;
    __nv_bfloat16 const* w_q_t;
    __nv_bfloat16 const* w_k_t;
    __nv_bfloat16 const* w_v_t;
    __nv_bfloat16 const* bias_q;
    __nv_bfloat16 const* bias_k;
    __nv_bfloat16 const* bias_v;
    __nv_bfloat16 const* cs_q;
    __nv_bfloat16 const* cs_k;
    __nv_bfloat16 const* cs_v;
    float const* a_log;
    __nv_bfloat16 const* g;
    float const* dt_bias;
    __nv_bfloat16 const* beta;
    __nv_bfloat16 const* onorm_g;
    float const* onorm_weight;
    float* state;
    __nv_bfloat16* out;
    int batch;
};

__device__ __forceinline__ float bf16_load(__nv_bfloat16 const* ptr, int index)
{
    return __bfloat162float(ptr[index]);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float value)
{
    return __float2bfloat16(value);
}

__device__ __forceinline__ float sigmoid_fast(float value)
{
    return 1.0f / (1.0f + __expf(-value));
}

__device__ __forceinline__ float silu_fast(float value)
{
    return value * sigmoid_fast(value);
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        x += __shfl_xor_sync(0xffffffffu, x, offset);
        y += __shfl_xor_sync(0xffffffffu, y, offset);
    }
    return {x, y};
}

__device__ __forceinline__ Sum2 block_reduce_sum_pair(float x, float y, float* scratch)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    float const warp_x = warp_reduce_sum(x);
    float const warp_y = warp_reduce_sum(y);
    if (lane == 0)
    {
        scratch[warp] = warp_x;
        scratch[kWarps + warp] = warp_y;
    }
    __syncthreads();

    float block_x = 0.0f;
    float block_y = 0.0f;
    if (warp == 0)
    {
        block_x = lane < kWarps ? scratch[lane] : 0.0f;
        block_y = lane < kWarps ? scratch[kWarps + lane] : 0.0f;
        block_x = warp_reduce_sum(block_x);
        block_y = warp_reduce_sum(block_y);
        // All lanes must finish reading partials before lane zero reuses scratch.
        __syncwarp();
        if (lane == 0)
        {
            scratch[0] = block_x;
            scratch[1] = block_y;
        }
    }
    __syncthreads();
    return {scratch[0], scratch[1]};
}

template <int kActiveWarps>
__device__ __forceinline__ float block_reduce_sum_active(float value, float* scratch)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    float warp_total = 0.0f;
    if (warp < kActiveWarps)
    {
        warp_total = warp_reduce_sum(value);
    }
    if (lane == 0 && warp < kActiveWarps)
    {
        scratch[warp] = warp_total;
    }
    __syncthreads();

    float block_total = 0.0f;
    if (warp == 0)
    {
        block_total = lane < kActiveWarps ? scratch[lane] : 0.0f;
        block_total = warp_reduce_sum(block_total);
        __syncwarp();
        if (lane == 0)
        {
            scratch[0] = block_total;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ void cp_async_16b(float* shared_ptr, float const* global_ptr)
{
    const uint32_t shared_address = static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(shared_address), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_one()
{
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_state_chunk(
    float* shared_state, float const* state, int batch_index, int head, int chunk, int stage)
{
    constexpr int kFloat4PerChunk = kChunkRows * kDim / 4;
    int const row_base = chunk * kChunkRows;
    for (int linear4 = threadIdx.x; linear4 < kFloat4PerChunk; linear4 += kThreads)
    {
        int const element = linear4 * 4;
        int const row = element / kDim;
        int const column = element - row * kDim;
        float* destination = shared_state + (stage * kChunkRows + row) * kDim + column;
        float const* source = state + ((batch_index * kHeads + head) * kDim + row_base + row) * kDim + column;
        cp_async_16b(destination, source);
    }
    cp_async_commit();
}

template <KernelSchedule kSchedule>
__global__ __launch_bounds__(kThreads, 2) void kda_decode_native_kernel(__nv_bfloat16 const* __restrict__ x_q,
    __nv_bfloat16 const* __restrict__ x_k, __nv_bfloat16 const* __restrict__ x_v,
    __nv_bfloat16 const* __restrict__ w_q_t, __nv_bfloat16 const* __restrict__ w_k_t,
    __nv_bfloat16 const* __restrict__ w_v_t, __nv_bfloat16 const* __restrict__ bias_q,
    __nv_bfloat16 const* __restrict__ bias_k, __nv_bfloat16 const* __restrict__ bias_v,
    __nv_bfloat16 const* __restrict__ cs_q, __nv_bfloat16 const* __restrict__ cs_k,
    __nv_bfloat16 const* __restrict__ cs_v, float const* __restrict__ a_log, __nv_bfloat16 const* __restrict__ g,
    float const* __restrict__ dt_bias, __nv_bfloat16 const* __restrict__ beta,
    __nv_bfloat16 const* __restrict__ onorm_g, float const* __restrict__ onorm_weight, float* __restrict__ state,
    __nv_bfloat16* __restrict__ out, float lower_bound, float scale, float onorm_epsilon)
{
    constexpr bool kUseCpAsyncBulk = kSchedule == KernelSchedule::kSingleCtaTwoStageCpAsyncBulk
        || kSchedule == KernelSchedule::kSingleCtaFourStageCpAsyncBulk;
    constexpr int kCpAsyncBulkStageCount = kSchedule == KernelSchedule::kSingleCtaFourStageCpAsyncBulk ? kChunks : 2;
    constexpr int kClusterBlocks = kSchedule == KernelSchedule::kFourCtaThreadBlockClusterCpAsync ? 4 : 0;
    constexpr bool kUseCluster = kClusterBlocks != 0;
    constexpr int kClusterChunksPerBlock = kUseCluster ? kChunks / kClusterBlocks : kChunks;
    constexpr int kLocalVThreads = kUseCluster ? kClusterChunksPerBlock * kChunkRows : kDim;
    constexpr int kVThreadBase = kUseCluster ? kDim : 0;
    constexpr int kStateStages = kUseCpAsyncBulk ? kCpAsyncBulkStageCount : (kUseCluster ? kClusterChunksPerBlock : 2);
    constexpr int kRowsPerWarp = kChunkRows / kWarps;
    static_assert(!kUseCpAsyncBulk || !kUseCluster);
    static_assert(!kUseCluster || kClusterBlocks == kChunks);

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    const cg::cluster_group cluster = cg::this_cluster();
    int const cluster_rank = kUseCluster ? static_cast<int>(cluster.block_rank()) : 0;
    int const batch_index = kUseCluster ? blockIdx.y : blockIdx.x;
    int const head = kUseCluster ? blockIdx.z : blockIdx.y;
    int const head_offset = head * kDim;

    using StateStage = float[kChunkRows][kDim];
    using StateBarrier = cuda::barrier<cuda::thread_scope_block>;
    static_assert(sizeof(StateBarrier) == sizeof(uint64_t));
    __shared__ StateSharedStorage<kUseCpAsyncBulk, kStateStages> state_storage;
    __shared__ CpAsyncBulkBarrierStorage<kUseCpAsyncBulk> barrier_storage;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    StateStage* shared_state;
    if constexpr (kUseCpAsyncBulk)
    {
        shared_state = reinterpret_cast<StateStage*>(dynamic_shared);
    }
    else
    {
        shared_state = state_storage.data;
    }
    __shared__ float shared_q[kDim];
    __shared__ float shared_k[kDim];
    __shared__ float shared_decay[kDim];
    __shared__ float shared_v[kDim];
    __shared__ float shared_output[kDim];
    __shared__ float reduction[kThreads];
    __shared__ float shared_beta;
    float preloaded_gate = 0.0f;
    float preloaded_weight = 0.0f;

    if constexpr (kUseCpAsyncBulk)
    {
        constexpr int kChunkBytes = kChunkRows * kDim * sizeof(float);
        const cg::thread_block block = cg::this_thread_block();
        if (tid < kChunks)
        {
            auto* ready = reinterpret_cast<StateBarrier*>(&barrier_storage.ready[tid]);
            init(ready, kThreads);
        }
        block.sync();
        float const* state_head = state + (batch_index * kHeads + head) * kDim * kDim;
#pragma unroll
        for (int chunk = 0; chunk < kCpAsyncBulkStageCount; ++chunk)
        {
            cuda::memcpy_async(block, &shared_state[chunk][0][0], state_head + chunk * kChunkRows * kDim,
                cuda::aligned_size_t<16>(kChunkBytes), *reinterpret_cast<StateBarrier*>(&barrier_storage.ready[chunk]));
        }
    }
    else
    {
        int const first_chunk = cluster_rank * kClusterChunksPerBlock;
        cp_async_state_chunk(&shared_state[0][0][0], state, batch_index, head, first_chunk, 0);
    }

    if (tid < kDim)
    {
        int const column = tid;
        int const head_column = head_offset + column;
        int const cache_base = batch_index * kFlat * kConvCacheWidth + head_column;
        float const exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[head]) : 0.0f, 0);

        float q_accumulator = bf16_load(bias_q, head_column);
        float k_accumulator = bf16_load(bias_k, head_column);
#pragma unroll
        for (int width = 0; width < kConvCacheWidth; ++width)
        {
            int const cache_index = cache_base + width * kFlat;
            q_accumulator += bf16_load(cs_q, cache_index) * bf16_load(w_q_t, width * kFlat + head_column);
            k_accumulator += bf16_load(cs_k, cache_index) * bf16_load(w_k_t, width * kFlat + head_column);
        }
        int const token_index = (batch_index * kHeads + head) * kDim + column;
        q_accumulator += bf16_load(x_q, token_index) * bf16_load(w_q_t, (kKernelWidth - 1) * kFlat + head_column);
        k_accumulator += bf16_load(x_k, token_index) * bf16_load(w_k_t, (kKernelWidth - 1) * kFlat + head_column);
        shared_q[column] = silu_fast(q_accumulator);
        shared_k[column] = silu_fast(k_accumulator);

        float const gate = bf16_load(g, token_index) + dt_bias[head_column];
        shared_decay[column] = __expf(lower_bound * sigmoid_fast(exp_a * gate));
    }

    if (tid >= kVThreadBase && tid < kVThreadBase + kLocalVThreads)
    {
        int const local_row = tid - kVThreadBase;
        int const row = kUseCluster ? cluster_rank * kLocalVThreads + local_row : local_row;
        int const head_row = head_offset + row;
        int const cache_base = batch_index * kFlat * kConvCacheWidth + head_row;
        float v_accumulator = bf16_load(bias_v, head_row);
#pragma unroll
        for (int width = 0; width < kConvCacheWidth; ++width)
        {
            int const cache_index = cache_base + width * kFlat;
            v_accumulator += bf16_load(cs_v, cache_index) * bf16_load(w_v_t, width * kFlat + head_row);
        }
        int const token_index = (batch_index * kHeads + head) * kDim + row;
        v_accumulator += bf16_load(x_v, token_index) * bf16_load(w_v_t, (kKernelWidth - 1) * kFlat + head_row);
        shared_v[row] = silu_fast(v_accumulator);

        if constexpr (!kUseCluster)
        {
            preloaded_gate = sigmoid_fast(bf16_load(onorm_g, token_index));
            preloaded_weight = onorm_weight[row];
        }
    }

    if (tid == 0)
    {
        shared_beta = sigmoid_fast(bf16_load(beta, batch_index * kHeads + head));
    }
    __syncthreads();

    if constexpr (!kUseCpAsyncBulk && (!kUseCluster || kClusterChunksPerBlock > 1))
    {
        int const second_chunk = cluster_rank * kClusterChunksPerBlock + 1;
        cp_async_state_chunk(&shared_state[0][0][0], state, batch_index, head, second_chunk, 1);
    }

    float const q_square = tid < kDim ? shared_q[tid] * shared_q[tid] : 0.0f;
    float const k_square = tid < kDim ? shared_k[tid] * shared_k[tid] : 0.0f;
    const Sum2 qk_sum = block_reduce_sum_pair(q_square, k_square, reduction);
    if (tid < kDim)
    {
        shared_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
        shared_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
    }
    __syncthreads();

    int const column_base = lane * 4;
    const float4 q_vector = *reinterpret_cast<float4 const*>(shared_q + column_base);
    const float4 k_vector = *reinterpret_cast<float4 const*>(shared_k + column_base);
    const float4 decay_vector = *reinterpret_cast<float4 const*>(shared_decay + column_base);
    float const q_values[4] = {q_vector.x, q_vector.y, q_vector.z, q_vector.w};
    float const k_values[4] = {k_vector.x, k_vector.y, k_vector.z, k_vector.w};
    float const decay_values[4] = {decay_vector.x, decay_vector.y, decay_vector.z, decay_vector.w};
    float output_sum_square = 0.0f;

#pragma unroll
    for (int local_chunk = 0; local_chunk < kClusterChunksPerBlock; ++local_chunk)
    {
        int const chunk = kUseCluster ? cluster_rank * kClusterChunksPerBlock + local_chunk : local_chunk;
        if constexpr (kUseCpAsyncBulk)
        {
            reinterpret_cast<StateBarrier*>(&barrier_storage.ready[chunk % kCpAsyncBulkStageCount])->arrive_and_wait();
        }
        else if constexpr (kUseCluster && kClusterChunksPerBlock > 1)
        {
            if (local_chunk + 1 < kClusterChunksPerBlock)
            {
                cp_async_wait_group_one();
            }
            else
            {
                cp_async_wait_all();
            }
        }
        else if constexpr (!kUseCluster)
        {
            if (chunk + 1 < kChunks)
            {
                cp_async_wait_group_one();
            }
            else
            {
                cp_async_wait_all();
            }
        }
        else
        {
            cp_async_wait_all();
        }
        if constexpr (!kUseCpAsyncBulk)
        {
            // Every warp consumes only the rows copied by the same warp.
            __syncwarp();
        }

#pragma unroll
        for (int row_offset = 0; row_offset < kRowsPerWarp; row_offset += 2)
        {
            int const shared_row_a = warp + row_offset * kWarps;
            int const shared_row_b = warp + (row_offset + 1) * kWarps;
            int const row_a = chunk * kChunkRows + shared_row_a;
            int const row_b = chunk * kChunkRows + shared_row_b;
            int const stage
                = kUseCpAsyncBulk ? chunk % kCpAsyncBulkStageCount : (kUseCluster ? local_chunk : chunk & 1);
            const float4 raw_state_a
                = *reinterpret_cast<float4 const*>(&shared_state[stage][shared_row_a][column_base]);
            const float4 raw_state_b
                = *reinterpret_cast<float4 const*>(&shared_state[stage][shared_row_b][column_base]);
            float state_a[4] = {
                raw_state_a.x * decay_values[0],
                raw_state_a.y * decay_values[1],
                raw_state_a.z * decay_values[2],
                raw_state_a.w * decay_values[3],
            };
            float state_b[4] = {
                raw_state_b.x * decay_values[0],
                raw_state_b.y * decay_values[1],
                raw_state_b.z * decay_values[2],
                raw_state_b.w * decay_values[3],
            };
            float const state_k_a = state_a[0] * k_values[0] + state_a[1] * k_values[1] + state_a[2] * k_values[2]
                + state_a[3] * k_values[3];
            float const state_k_b = state_b[0] * k_values[0] + state_b[1] * k_values[1] + state_b[2] * k_values[2]
                + state_b[3] * k_values[3];
            const Sum2 state_k = warp_reduce_sum_pair(state_k_a, state_k_b);
            float const residual_a = (shared_v[row_a] - state_k.x) * shared_beta;
            float const residual_b = (shared_v[row_b] - state_k.y) * shared_beta;
#pragma unroll
            for (int component = 0; component < 4; ++component)
            {
                state_a[component] += k_values[component] * residual_a;
                state_b[component] += k_values[component] * residual_b;
            }

            int const state_index_a = ((batch_index * kHeads + head) * kDim + row_a) * kDim + column_base;
            int const state_index_b = ((batch_index * kHeads + head) * kDim + row_b) * kDim + column_base;
            *reinterpret_cast<float4*>(state + state_index_a)
                = make_float4(state_a[0], state_a[1], state_a[2], state_a[3]);
            *reinterpret_cast<float4*>(state + state_index_b)
                = make_float4(state_b[0], state_b[1], state_b[2], state_b[3]);

            float const state_q_a = state_a[0] * q_values[0] + state_a[1] * q_values[1] + state_a[2] * q_values[2]
                + state_a[3] * q_values[3];
            float const state_q_b = state_b[0] * q_values[0] + state_b[1] * q_values[1] + state_b[2] * q_values[2]
                + state_b[3] * q_values[3];
            const Sum2 state_q = warp_reduce_sum_pair(state_q_a, state_q_b);
            if (lane == 0)
            {
                shared_output[row_a] = state_q.x;
                shared_output[row_b] = state_q.y;
                if constexpr (kUseCluster)
                {
                    output_sum_square += state_q.x * state_q.x + state_q.y * state_q.y;
                }
            }
        }

        if constexpr (kUseCpAsyncBulk)
        {
            int const next_chunk = chunk + kCpAsyncBulkStageCount;
            if (next_chunk < kChunks)
            {
                constexpr int kChunkBytes = kChunkRows * kDim * sizeof(float);
                int const stage = chunk % kCpAsyncBulkStageCount;
                const cg::thread_block block = cg::this_thread_block();
                // A stage cannot be refilled until every warp has consumed it.
                block.sync();
                float const* next_state
                    = state + ((batch_index * kHeads + head) * kDim + next_chunk * kChunkRows) * kDim;
                cuda::memcpy_async(block, &shared_state[stage][0][0], next_state, cuda::aligned_size_t<16>(kChunkBytes),
                    *reinterpret_cast<StateBarrier*>(&barrier_storage.ready[stage]));
            }
        }
        else if constexpr (!kUseCluster)
        {
            if (chunk + 2 < kChunks)
            {
                cp_async_state_chunk(&shared_state[0][0][0], state, batch_index, head, chunk + 2, (chunk + 2) & 1);
            }
        }
    }
    __syncthreads();

    if constexpr (kUseCluster)
    {
        if (lane == 0)
        {
            reduction[warp] = output_sum_square;
        }
        __syncthreads();

        float block_sum_square = 0.0f;
        if (warp == 0)
        {
            block_sum_square = lane < kWarps ? reduction[lane] : 0.0f;
            block_sum_square = warp_reduce_sum(block_sum_square);
            if (lane == 0)
            {
                reduction[0] = block_sum_square;
            }
        }
        __syncthreads();

        cluster.sync();
        if (tid == 0 && cluster_rank == 0)
        {
            float cluster_sum_square = 0.0f;
#pragma unroll
            for (int rank = 0; rank < kClusterBlocks; ++rank)
            {
                cluster_sum_square += *cluster.map_shared_rank(&reduction[0], rank);
            }
            reduction[0] = cluster_sum_square;
        }
        cluster.sync();
        float const normalization_sum_square = *cluster.map_shared_rank(&reduction[0], 0);

        if (tid >= kVThreadBase && tid < kVThreadBase + kLocalVThreads)
        {
            int const local_row = tid - kVThreadBase;
            int const row = cluster_rank * kLocalVThreads + local_row;
            int const output_index = (batch_index * kHeads + head) * kDim + row;
            float const inverse_rms = rsqrtf(normalization_sum_square / static_cast<float>(kDim) + onorm_epsilon);
            float const output_value
                = shared_output[row] * inverse_rms * onorm_weight[row] * sigmoid_fast(bf16_load(onorm_g, output_index));
            out[output_index] = bf16_store(output_value);
        }
        // Rank zero's shared total must remain live until every peer consumes it.
        cluster.sync();
    }
    else
    {
        float const raw_output = tid < kDim ? shared_output[tid] : 0.0f;
        float const normalization_sum_square = block_reduce_sum_active<kDim / 32>(raw_output * raw_output, reduction);
        if (tid < kDim)
        {
            int const output_index = (batch_index * kHeads + head) * kDim + tid;
            float const inverse_rms = rsqrtf(normalization_sum_square / static_cast<float>(kDim) + onorm_epsilon);
            out[output_index] = bf16_store(raw_output * inverse_rms * preloaded_weight * preloaded_gate);
        }
    }
}

template <KernelSchedule kSchedule>
cudaError_t launch_kernel_schedule(KernelArguments const& arguments, cudaStream_t stream)
{
    constexpr bool kUseCpAsyncBulk = kSchedule == KernelSchedule::kSingleCtaTwoStageCpAsyncBulk
        || kSchedule == KernelSchedule::kSingleCtaFourStageCpAsyncBulk;
    constexpr int kCpAsyncBulkStageCount = kSchedule == KernelSchedule::kSingleCtaFourStageCpAsyncBulk ? kChunks : 2;
    constexpr int kClusterBlocks = kSchedule == KernelSchedule::kFourCtaThreadBlockClusterCpAsync ? 4 : 0;
    constexpr bool kUseCluster = kClusterBlocks != 0;
    constexpr int kDynamicSharedBytes
        = kUseCpAsyncBulk ? kCpAsyncBulkStageCount * kChunkRows * kDim * sizeof(float) : 0;

    if constexpr (kUseCpAsyncBulk)
    {
        const cudaError_t attribute_status = cudaFuncSetAttribute(
            kda_decode_native_kernel<kSchedule>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedBytes);
        if (attribute_status != cudaSuccess)
        {
            return attribute_status;
        }
    }

    if constexpr (kUseCluster)
    {
        cudaLaunchAttribute cluster_attribute{};
        cluster_attribute.id = cudaLaunchAttributeClusterDimension;
        cluster_attribute.val.clusterDim.x = kClusterBlocks;
        cluster_attribute.val.clusterDim.y = 1;
        cluster_attribute.val.clusterDim.z = 1;

        cudaLaunchConfig_t config{};
        config.gridDim = dim3(kClusterBlocks, arguments.batch, kHeads);
        config.blockDim = dim3(kThreads);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        config.attrs = &cluster_attribute;
        config.numAttrs = 1;

        const cudaError_t launch_status = cudaLaunchKernelEx(&config, kda_decode_native_kernel<kSchedule>,
            arguments.x_q, arguments.x_k, arguments.x_v, arguments.w_q_t, arguments.w_k_t, arguments.w_v_t,
            arguments.bias_q, arguments.bias_k, arguments.bias_v, arguments.cs_q, arguments.cs_k, arguments.cs_v,
            arguments.a_log, arguments.g, arguments.dt_bias, arguments.beta, arguments.onorm_g, arguments.onorm_weight,
            arguments.state, arguments.out, kLowerBound, kQScale, kOutputNormEpsilon);
        if (launch_status != cudaSuccess)
        {
            return launch_status;
        }
        return cudaGetLastError();
    }

    kda_decode_native_kernel<kSchedule><<<dim3(arguments.batch, kHeads), dim3(kThreads), kDynamicSharedBytes, stream>>>(
        arguments.x_q, arguments.x_k, arguments.x_v, arguments.w_q_t, arguments.w_k_t, arguments.w_v_t,
        arguments.bias_q, arguments.bias_k, arguments.bias_v, arguments.cs_q, arguments.cs_k, arguments.cs_v,
        arguments.a_log, arguments.g, arguments.dt_bias, arguments.beta, arguments.onorm_g, arguments.onorm_weight,
        arguments.state, arguments.out, kLowerBound, kQScale, kOutputNormEpsilon);
    return cudaGetLastError();
}

bool has_null_pointer(KernelArguments const& arguments)
{
    return arguments.x_q == nullptr || arguments.x_k == nullptr || arguments.x_v == nullptr
        || arguments.w_q_t == nullptr || arguments.w_k_t == nullptr || arguments.w_v_t == nullptr
        || arguments.bias_q == nullptr || arguments.bias_k == nullptr || arguments.bias_v == nullptr
        || arguments.cs_q == nullptr || arguments.cs_k == nullptr || arguments.cs_v == nullptr
        || arguments.a_log == nullptr || arguments.g == nullptr || arguments.dt_bias == nullptr
        || arguments.beta == nullptr || arguments.onorm_g == nullptr || arguments.onorm_weight == nullptr
        || arguments.state == nullptr || arguments.out == nullptr;
}

} // namespace

extern "C" cudaError_t launch_kda_decode_native_cuda(void const* x_q, void const* x_k, void const* x_v,
    void const* w_q_t, void const* w_k_t, void const* w_v_t, void const* bias_q, void const* bias_k, void const* bias_v,
    void const* cs_q, void const* cs_k, void const* cs_v, float const* a_log, void const* g, float const* dt_bias,
    void const* beta, void const* onorm_g, float const* onorm_weight, float* state, void* out, int batch, int schedule,
    cudaStream_t stream)
{
    const KernelArguments arguments{
        reinterpret_cast<__nv_bfloat16 const*>(x_q),
        reinterpret_cast<__nv_bfloat16 const*>(x_k),
        reinterpret_cast<__nv_bfloat16 const*>(x_v),
        reinterpret_cast<__nv_bfloat16 const*>(w_q_t),
        reinterpret_cast<__nv_bfloat16 const*>(w_k_t),
        reinterpret_cast<__nv_bfloat16 const*>(w_v_t),
        reinterpret_cast<__nv_bfloat16 const*>(bias_q),
        reinterpret_cast<__nv_bfloat16 const*>(bias_k),
        reinterpret_cast<__nv_bfloat16 const*>(bias_v),
        reinterpret_cast<__nv_bfloat16 const*>(cs_q),
        reinterpret_cast<__nv_bfloat16 const*>(cs_k),
        reinterpret_cast<__nv_bfloat16 const*>(cs_v),
        a_log,
        reinterpret_cast<__nv_bfloat16 const*>(g),
        dt_bias,
        reinterpret_cast<__nv_bfloat16 const*>(beta),
        reinterpret_cast<__nv_bfloat16 const*>(onorm_g),
        onorm_weight,
        state,
        reinterpret_cast<__nv_bfloat16*>(out),
        batch,
    };
    if (batch <= 0 || has_null_pointer(arguments))
    {
        return cudaErrorInvalidValue;
    }

    switch (static_cast<KernelSchedule>(schedule))
    {
    case KernelSchedule::kSingleCtaCpAsync:
        return launch_kernel_schedule<KernelSchedule::kSingleCtaCpAsync>(arguments, stream);
    case KernelSchedule::kSingleCtaTwoStageCpAsyncBulk:
        return launch_kernel_schedule<KernelSchedule::kSingleCtaTwoStageCpAsyncBulk>(arguments, stream);
    case KernelSchedule::kSingleCtaFourStageCpAsyncBulk:
        return launch_kernel_schedule<KernelSchedule::kSingleCtaFourStageCpAsyncBulk>(arguments, stream);
    case KernelSchedule::kFourCtaThreadBlockClusterCpAsync:
        return launch_kernel_schedule<KernelSchedule::kFourCtaThreadBlockClusterCpAsync>(arguments, stream);
    default: return cudaErrorInvalidValue;
    }
}
