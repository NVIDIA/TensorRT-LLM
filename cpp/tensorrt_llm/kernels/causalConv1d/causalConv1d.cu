
/*
 * Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu
 * and https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu
 * Copyright (c) 2024, Tri Dao.
 *
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "tensorrt_llm/kernels/causalConv1d/causalConv1d.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::causal_conv1d
{

template <int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_fwd_kernel_traits
{
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static_assert(kNThreads_ % 32 == 0, "kNThreads must be a multiple of 32 for warp shuffle");
    static_assert(sizeof(vec_t) == 16, "vec_t must be 16 bytes for warp shuffle optimization");
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template <typename Ktraits, bool kHasConvStateIndices, bool kSiluActivation>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_fwd_kernel(ConvParamsBase params)
{
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_ + Ktraits::kSmemIOSize);

    bool const kVarlen = params.query_start_loc_ptr != nullptr;
    int const tidx = threadIdx.x;
    int const batch_id = blockIdx.x;
    int const channel_id = blockIdx.y;
    int const* query_start_loc = kVarlen ? reinterpret_cast<int*>(params.query_start_loc_ptr) : nullptr;
    int const sequence_start_index = kVarlen ? query_start_loc[batch_id] : batch_id;
    int const seqlen = kVarlen ? query_start_loc[batch_id + 1] - sequence_start_index : params.seqlen;

    input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + sequence_start_index * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t* weight = reinterpret_cast<weight_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + sequence_start_index * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t*>(params.bias_ptr)[channel_id]);

    bool has_initial_state = params.has_initial_state_ptr == nullptr
        ? false
        : reinterpret_cast<bool*>(params.has_initial_state_ptr)[batch_id];

    int cache_index;
    if constexpr (kHasConvStateIndices)
    {
        cache_index = reinterpret_cast<int*>(params.cache_indices_ptr)[batch_id];
        if (cache_index == params.pad_slot_id)
        {
            return;
        }
    }
    else
    {
        cache_index = batch_id;
    }
    input_t* conv_states = params.conv_states_ptr == nullptr ? nullptr
                                                             : reinterpret_cast<input_t*>(params.conv_states_ptr)
            + cache_index * params.conv_states_batch_stride + channel_id * params.conv_states_c_stride;

    // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
    if (tidx == 0)
    {
        input_t initial_state[kNElts] = {0};
        if (has_initial_state)
        {
#pragma unroll
            for (int w = 0; w < kWidth - 1; ++w)
            {
                initial_state[kNElts - 1 - (kWidth - 2) + w] = conv_states[w];
            }
        }
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t*>(initial_state)[0];
    }

    // Save final conv_state from the tail of x directly, instead of reconstructing it
    // from smem_exchange after the main loop.
    if (conv_states != nullptr && tidx == 0)
    {
        if (seqlen >= kWidth - 1)
        {
#pragma unroll
            for (int w = 0; w < kWidth - 1; ++w)
            {
                conv_states[w] = x[(seqlen - (kWidth - 1) + w) * params.x_l_stride];
            }
        }
        else
        {
#pragma unroll
            for (int w = 0; w < kWidth - 1; ++w)
            {
                if (w < (kWidth - 1) - seqlen)
                {
                    conv_states[w] = has_initial_state ? conv_states[w + seqlen] : input_t(0.0f);
                }
                else
                {
                    conv_states[w] = x[(w - ((kWidth - 1) - seqlen)) * params.x_l_stride];
                }
            }
        }
    }

    float weight_vals[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth; ++i)
    {
        weight_vals[i] = float(__ldg(&weight[i * params.weight_width_stride]));
    }

    constexpr int kChunkSize = kNThreads * kNElts;
    int const n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk)
    {
        input_t x_vals_load[2 * kNElts] = {0};
        if constexpr (kIsVecLoad)
        {
            typename Ktraits::BlockLoadVecT(smem_load_vec)
                .Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t(*)[1]>(&x_vals_load[kNElts]),
                    (seqlen - chunk * kChunkSize) / kNElts);
        }
        else
        {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(
                x, *reinterpret_cast<input_t(*)[kNElts]>(&x_vals_load[kNElts]), seqlen - chunk * kChunkSize);
        }
        x += kChunkSize;

        int const lane_id = tidx & 31;
        vec_t high_val = reinterpret_cast<vec_t*>(x_vals_load)[1];

        __syncthreads();
        // Thread kNThreads - 1 don't write yet, so that thread 0 can read
        // the last elements of the previous chunk.
        if (tidx < kNThreads - 1)
        {
            smem_exchange[tidx] = high_val;
        }
        __syncthreads();

        // Get neighbor data: use warp shuffle for most threads, shared memory for warp boundaries
        vec_t neighbor;
        uint32_t* high_val_p = reinterpret_cast<uint32_t*>(&high_val);
        uint32_t* nbr_p = reinterpret_cast<uint32_t*>(&neighbor);
        nbr_p[0] = __shfl_up_sync(0xFFFFFFFF, high_val_p[0], 1);
        nbr_p[1] = __shfl_up_sync(0xFFFFFFFF, high_val_p[1], 1);
        nbr_p[2] = __shfl_up_sync(0xFFFFFFFF, high_val_p[2], 1);
        nbr_p[3] = __shfl_up_sync(0xFFFFFFFF, high_val_p[3], 1);

        // Lane 0 must use shared memory to handle the cross-warp boundary.
        // thread 0 uses the last element of the previous chunk.
        if (lane_id == 0)
        {
            neighbor = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        }
        reinterpret_cast<vec_t*>(x_vals_load)[0] = neighbor;

        __syncthreads();
        // Now thread kNThreads - 1 can write the last elements of the current chunk.
        if (tidx == kNThreads - 1)
        {
            smem_exchange[tidx] = high_val;
        }

        float x_vals[2 * kNElts];
#pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i)
        {
            x_vals[i] = float(x_vals_load[i]);
        }

        float out_vals[kNElts];
#pragma unroll
        // Process 2 outputs at a time for better ILP (instruction level parallelism).
        for (int i = 0; i < kNElts; i += 2)
        {
            float acc0 = bias_val;
            float acc1 = bias_val;
#pragma unroll
            for (int w = 0; w < kWidth; ++w)
            {
                float wt = weight_vals[w];
                acc0 = __fmaf_rn(wt, x_vals[kNElts + i - (kWidth - w - 1)], acc0);
                acc1 = __fmaf_rn(wt, x_vals[kNElts + i + 1 - (kWidth - w - 1)], acc1);
            }
            out_vals[i] = acc0;
            out_vals[i + 1] = acc1;
        }

        if constexpr (kSiluActivation)
        {
#pragma unroll
            for (int i = 0; i < kNElts; i += 2)
            {
                // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                // Using fast math: __expf and __frcp_rn
                float v0 = out_vals[i];
                float v1 = out_vals[i + 1];
                out_vals[i] = v0 * __frcp_rn(1.0f + __expf(-v0));
                out_vals[i + 1] = v1 * __frcp_rn(1.0f + __expf(-v1));
            }
        }

        input_t out_vals_store[kNElts];
#pragma unroll
        for (int i = 0; i < kNElts; ++i)
        {
            out_vals_store[i] = out_vals[i];
        }
        if constexpr (kIsVecLoad)
        {
            typename Ktraits::BlockStoreVecT(smem_store_vec)
                .Store(reinterpret_cast<vec_t*>(out), reinterpret_cast<vec_t(&)[1]>(out_vals_store),
                    (seqlen - chunk * kChunkSize) / kNElts);
        }
        else
        {
            typename Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;
    }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase& params, cudaStream_t stream)
{
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    bool const kVarlen = params.query_start_loc_ptr != nullptr;
    // Enable vectorized 128-bit loads when total tokens are aligned. For varlen with
    // batch==1 (common prefill), seq_start is always 0 so alignment is guaranteed.
    bool const canVecLoad = params.seqlen % kNElts == 0 && (!kVarlen || params.batch == 1);
    BOOL_SWITCH(canVecLoad, kIsVecLoad,
        [&]
        {
            using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize;
            dim3 grid(params.batch, params.dim);
            bool const hasConvStateIdx = params.cache_indices_ptr != nullptr;
            BOOL_SWITCH(hasConvStateIdx, kHasCSI,
                [&]
                {
                    BOOL_SWITCH(params.silu_activation, kSilu,
                        [&]
                        {
                            auto kernel = &causal_conv1d_fwd_kernel<Ktraits, kHasCSI, kSilu>;
                            if (kSmemSize >= 48 * 1024)
                            {
                                TLLM_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            }
                            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                        });
                });
            TLLM_CUDA_KERNEL_LAUNCH_CHECK();
        });
}

template <int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_dispatch(ConvParamsBase& params, cudaStream_t stream)
{
    bool const isVarlen = params.query_start_loc_ptr != nullptr;
    constexpr int kNarrowThreads = 64;
    constexpr int kWideThreads = 128;
    constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    constexpr int kShortSeqThreshold = kNarrowThreads * kNElts;
    // Pick the wider 128-thread kernel when the average per-sequence length exceeds
    // one chunk; otherwise the narrower 64-thread kernel avoids overprovisioning.
    int const avgSeqlen = isVarlen ? (params.seqlen / max(params.batch, 1)) : params.seqlen;
    bool const preferNarrowKernel = avgSeqlen <= kShortSeqThreshold;

    if (preferNarrowKernel)
    {
        causal_conv1d_fwd_launch<kNarrowThreads, kWidth, input_t, weight_t>(params, stream);
    }
    else
    {
        causal_conv1d_fwd_launch<kWideThreads, kWidth, input_t, weight_t>(params, stream);
    }
}

template <typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase& params, cudaStream_t stream)
{
    if (params.width == 2)
    {
        causal_conv1d_fwd_dispatch<2, input_t, weight_t>(params, stream);
    }
    else if (params.width == 3)
    {
        causal_conv1d_fwd_dispatch<3, input_t, weight_t>(params, stream);
    }
    else if (params.width == 4)
    {
        causal_conv1d_fwd_dispatch<4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_fwd_cuda<float, float>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<half, half>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<nv_bfloat16, nv_bfloat16>(ConvParamsBase& params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Channel-last (token-major) causal conv1d forward.
//
// Layout contract: x and out have unit stride along the *channel* axis and a large stride along
// the token axis (x_c_stride == out_c_stride == 1).  This is the layout produced directly by a
// [tokens, channels] projection, so callers no longer have to transpose into the channel-major
// layout that `causal_conv1d_fwd_kernel` requires and transpose the result back.
//
// Decomposition: one thread owns kEltsPerThread consecutive channels (one 16B vector) and walks
// the token axis, carrying a (kWidth-1)-deep sliding window of *its own* channels in registers.
// No shared memory and no cross-thread communication are needed, so the memory access pattern
// degenerates to a strided streaming copy - consecutive threads touch consecutive 16B chunks of
// the same token row.  A block covers kNThreads*kEltsPerThread channels x kChunkSizeL tokens.
//
// NOTE: this kernel cannot run in-place.  Chunk k reads a (kWidth-1)-token halo written by chunk
// k-1, so aliasing out onto x would race across blocks.  The dispatcher enforces out != x.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t_, typename weight_t_, int kNThreads_, int kWidth_, int kEltsPerThread_, int kChunkSizeL_,
    int kUnroll_, int kSiluMode_>
struct Causal_conv1d_channellast_fwd_kernel_traits
{
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kEltsPerThread = kEltsPerThread_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kUnroll = kUnroll_;
    static constexpr int kSiluMode = kSiluMode_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static_assert(sizeof(weight_t) == sizeof(input_t));
    using vec_t = typename BytesToType<kNBytes * kEltsPerThread>::Type;
    static_assert(kChunkSizeL % kUnroll_ == 0, "chunk size must be a multiple of the unroll factor");
    static_assert(kNThreads_ % 32 == 0);
};

// Pre-activation threshold below which the tanh form of SiLU is no longer accurate to one ulp
// of the 16-bit result: silu(v) = h + h*tanh(h) amplifies the tanh absolute error by
// 1/(1+tanh(h)) as tanh(h) -> -1, so it drifts for deep-negative pre-activations (bf16 even
// flushes to +0 below v ~ -13).  A 16-bit-grid sweep is clean above -7.5; -7.0 adds margin.
static constexpr float kChannelLastTanhSiluCutoff = -7.0f;

// SiLU flavours.  Mode 1 is the default (see getCausalConv1dSiluMode); modes 2 and 3 are
// opt-in experiments behind TRTLLM_CAUSAL_CONV1D_SILU_MODE and are never selected on their own.
//   kMode 1: v * __frcp_rn(1 + __expf(-v)).  Bit-identical to causal_conv1d_fwd_kernel, but
//            __frcp_rn is the IEEE correctly-rounded reciprocal and expands to ~10 instructions.
//   kMode 2: __fdividef, i.e. rcp.approx.  fp32 relative error ~1e-6, well under one bf16 ulp.
//   kMode 3: 16-bit output only.  silu(v) = h + h*tanh.approx(h), h = v/2, a single MUFU
//            op instead of ex2+rcp.  The tanh absolute error is amplified by 1/(1+tanh(h)) as
//            tanh(h) -> -1, so it is only trusted above kChannelLastTanhSiluCutoff; a warp vote
//            reroutes the whole warp to the exact form otherwise (see causal_conv1d_silu_vec).
template <int kMode>
__device__ __forceinline__ float causal_conv1d_silu(float v)
{
    if constexpr (kMode == 2)
    {
        return __fdividef(v, 1.0f + __expf(-v));
    }
    else if constexpr (kMode == 3)
    {
        float const h = 0.5f * v;
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
        return __fmaf_rn(h, t, h);
    }
    else
    {
        return v * __frcp_rn(1.0f + __expf(-v));
    }
}

//! Applies the activation to one thread's kN accumulators.
//!
//! For kMode 3 the deep-negative region is handled by voting the whole warp onto the exact
//! formula: both branches are correct for every input, the vote only keeps the warp
//! non-divergent, and deep-negative pre-activations are rare enough that the exact path
//! essentially never runs.  `warpMask` must contain exactly the lanes that reach this point.
template <int kMode, int kN>
__device__ __forceinline__ void causal_conv1d_silu_vec(float (&v)[kN], unsigned int warpMask)
{
    if constexpr (kMode == 0)
    {
        return;
    }
    else if constexpr (kMode == 3)
    {
        float lo = v[0];
#pragma unroll
        for (int i = 1; i < kN; ++i)
        {
            lo = fminf(lo, v[i]);
        }
        if (!__any_sync(warpMask, lo <= kChannelLastTanhSiluCutoff))
        {
#pragma unroll
            for (int i = 0; i < kN; ++i)
            {
                v[i] = causal_conv1d_silu<3>(v[i]);
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            v[i] = causal_conv1d_silu<1>(v[i]);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            v[i] = causal_conv1d_silu<kMode>(v[i]);
        }
    }
}

// No minBlocksPerSM: ptxas spends 127 registers per thread here and spills nothing, which caps
// residency at 4 blocks/SM (~23% of the warp slots).  That looks like an occupancy problem but is
// not one - every sliding-window value has to stay live across the token loop, so forcing more
// resident blocks only converts registers into local-memory traffic on a kernel that is already
// moving its input at 3.7 TB/s.  Measured on B300 at dim 6144 / T 32768 / bf16 / width 4, against
// 220us unconstrained: 80 regs (6 blocks, 40-72B spill) 297-404us, 64 regs (8 blocks, 136-160B)
// 364-514us, 48 regs (10 blocks, 240-288B) 653us - monotonically worse with spill size.
template <typename Ktraits, bool kHasCacheIndices>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_channellast_fwd_kernel(ConvParamsBase params)
{
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using vec_t = typename Ktraits::vec_t;
    constexpr int kW = Ktraits::kWidth;
    constexpr int kE = Ktraits::kEltsPerThread;
    constexpr int kL = Ktraits::kChunkSizeL;
    constexpr int kU = Ktraits::kUnroll;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kSilu = Ktraits::kSiluMode;

    int const batch_id = blockIdx.z;
    int const c0 = (blockIdx.x * kNThreads + threadIdx.x) * kE;
    // Taken while the warp is still fully converged; every later exit is block-uniform, so this
    // is exactly the set of lanes that reaches the activation vote.
    unsigned int const warpMask = __ballot_sync(0xffffffffu, c0 < params.dim);
    if (c0 >= params.dim)
    {
        return;
    }

    int const* query_start_loc = reinterpret_cast<int const*>(params.query_start_loc_ptr);
    bool const kVarlen = query_start_loc != nullptr;
    int const sequence_start_index = kVarlen ? query_start_loc[batch_id] : batch_id;
    int const seqlen = kVarlen ? query_start_loc[batch_id + 1] - sequence_start_index : params.seqlen;

    int cache_index = batch_id;
    if constexpr (kHasCacheIndices)
    {
        cache_index = reinterpret_cast<int const*>(params.cache_indices_ptr)[batch_id];
        if (cache_index == params.pad_slot_id)
        {
            return;
        }
    }

    bool const has_initial_state = params.has_initial_state_ptr == nullptr
        ? false
        : reinterpret_cast<bool const*>(params.has_initial_state_ptr)[batch_id];

    int64_t const x_l = static_cast<int64_t>(params.x_l_stride);
    int64_t const o_l = static_cast<int64_t>(params.out_l_stride);

    input_t const* __restrict__ x = reinterpret_cast<input_t const*>(params.x_ptr)
        + static_cast<int64_t>(sequence_start_index) * static_cast<int64_t>(params.x_batch_stride) + c0;
    input_t* __restrict__ out = reinterpret_cast<input_t*>(params.out_ptr)
        + static_cast<int64_t>(sequence_start_index) * static_cast<int64_t>(params.out_batch_stride) + c0;

    int64_t const cs_c = static_cast<int64_t>(params.conv_states_c_stride);
    int64_t const cs_l = static_cast<int64_t>(params.conv_states_l_stride);
    input_t* conv_states = params.conv_states_ptr == nullptr ? nullptr
                                                             : reinterpret_cast<input_t*>(params.conv_states_ptr)
            + static_cast<int64_t>(cache_index) * static_cast<int64_t>(params.conv_states_batch_stride) + c0 * cs_c;

    {
        int const chunk_l_id = blockIdx.y;
        int const tok0 = chunk_l_id * kL;
        int const seg = min(kL, seqlen - tok0);
        // Chunks past the end of *this* sequence have no work.  Bail out before touching x: the
        // chunk grid is sized from the total token count, which overshoots for short sequences.
        // chunk 0 still runs so that conv_states is published even for an empty sequence.
        if (seg <= 0 && chunk_l_id != 0)
        {
            return;
        }

        // Sliding window over the token axis; xw[0 .. kW-2] is the causal carry.
        float xw[kW - 1 + kU][kE];

        // 1. Seed the causal window.  Must read conv_states *before* step 2 overwrites them.
        if (chunk_l_id == 0)
        {
            if (has_initial_state && conv_states != nullptr)
            {
#pragma unroll
                for (int w = 0; w < kW - 1; ++w)
                {
#pragma unroll
                    for (int e = 0; e < kE; ++e)
                    {
                        xw[w][e] = float(conv_states[e * cs_c + w * cs_l]);
                    }
                }
            }
            else
            {
#pragma unroll
                for (int w = 0; w < kW - 1; ++w)
                {
#pragma unroll
                    for (int e = 0; e < kE; ++e)
                    {
                        xw[w][e] = 0.f;
                    }
                }
            }
        }
        else
        {
            input_t const* xh = x + static_cast<int64_t>(tok0 - (kW - 1)) * x_l;
#pragma unroll
            for (int w = 0; w < kW - 1; ++w)
            {
                vec_t hv = *reinterpret_cast<vec_t const*>(xh + static_cast<int64_t>(w) * x_l);
                input_t const* hp = reinterpret_cast<input_t const*>(&hv);
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    xw[w][e] = float(hp[e]);
                }
            }
        }

        // 2. Publish the updated conv_states straight from the tail of x (chunk 0 owns this).
        if (conv_states != nullptr && chunk_l_id == 0)
        {
            if (seqlen >= kW - 1)
            {
                input_t const* xt = x + static_cast<int64_t>(seqlen - (kW - 1)) * x_l;
#pragma unroll
                for (int w = 0; w < kW - 1; ++w)
                {
#pragma unroll
                    for (int e = 0; e < kE; ++e)
                    {
                        conv_states[e * cs_c + w * cs_l] = xt[static_cast<int64_t>(w) * x_l + e];
                    }
                }
            }
            else
            {
#pragma unroll
                for (int w = 0; w < kW - 1; ++w)
                {
#pragma unroll
                    for (int e = 0; e < kE; ++e)
                    {
                        input_t v;
                        if (w < (kW - 1) - seqlen)
                        {
                            v = has_initial_state ? conv_states[e * cs_c + (w + seqlen) * cs_l] : input_t(0.f);
                        }
                        else
                        {
                            v = x[static_cast<int64_t>(w - ((kW - 1) - seqlen)) * x_l + e];
                        }
                        conv_states[e * cs_c + w * cs_l] = v;
                    }
                }
            }
        }

        if (seg <= 0)
        {
            return;
        }

        // ---- Per-channel weights / bias.  Block invariant, so loaded after the empty-chunk early-out. ----
        //
        // These are easy to get catastrophically wrong: a per-(channel, tap) scalar load makes each
        // warp touch kE*kWidth distinct 32B sectors per instruction (neighbouring threads are kE
        // channels apart), which on a 6144-channel shape issues as many L1 sector requests as the
        // entire x stream and doubles the kernel's L1 traffic.  For the usual dense (dim, width)
        // weight layout each thread's kE*kWidth taps are contiguous, so pull them in vectorised.
        weight_t const* wp
            = reinterpret_cast<weight_t const*>(params.weight_ptr) + static_cast<int64_t>(c0) * params.weight_c_stride;
        float wt[kW][kE];
        constexpr int kWBytes = kE * kW * sizeof(weight_t);
        constexpr int kWChunk = (kWBytes % 16 == 0) ? 16 : ((kWBytes % 8 == 0) ? 8 : ((kWBytes % 4 == 0) ? 4 : 2));
        using wvec_t = typename BytesToType<kWChunk>::Type;
        if (params.weight_width_stride == 1 && params.weight_c_stride == kW
            && (reinterpret_cast<uintptr_t>(wp) % kWChunk) == 0)
        {
            wvec_t wbuf[kWBytes / kWChunk];
#pragma unroll
            for (int i = 0; i < kWBytes / kWChunk; ++i)
            {
                wbuf[i] = reinterpret_cast<wvec_t const*>(wp)[i];
            }
            weight_t const* wraw = reinterpret_cast<weight_t const*>(&wbuf[0]);
#pragma unroll
            for (int e = 0; e < kE; ++e)
            {
#pragma unroll
                for (int w = 0; w < kW; ++w)
                {
                    wt[w][e] = float(wraw[e * kW + w]);
                }
            }
        }
        else
        {
#pragma unroll
            for (int e = 0; e < kE; ++e)
            {
#pragma unroll
                for (int w = 0; w < kW; ++w)
                {
                    wt[w][e] = float(
                        __ldg(&wp[static_cast<int64_t>(e) * params.weight_c_stride + w * params.weight_width_stride]));
                }
            }
        }

        float bias_v[kE];
        if (params.bias_ptr == nullptr)
        {
#pragma unroll
            for (int e = 0; e < kE; ++e)
            {
                bias_v[e] = 0.f;
            }
        }
        else
        {
            weight_t const* bp = reinterpret_cast<weight_t const*>(params.bias_ptr) + c0;
            if ((reinterpret_cast<uintptr_t>(bp) % sizeof(vec_t)) == 0)
            {
                vec_t bvec = *reinterpret_cast<vec_t const*>(bp);
                weight_t const* braw = reinterpret_cast<weight_t const*>(&bvec);
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    bias_v[e] = float(braw[e]);
                }
            }
            else
            {
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    bias_v[e] = float(__ldg(&bp[e]));
                }
            }
        }

        // 3. Stream the chunk.
        input_t const* xp = x + static_cast<int64_t>(tok0) * x_l;
        input_t* op = out + static_cast<int64_t>(tok0) * o_l;
        int t = 0;
        int const nfull = (seg / kU) * kU;
        for (; t < nfull; t += kU)
        {
            vec_t v[kU];
#pragma unroll
            for (int u = 0; u < kU; ++u)
            {
                v[u] = *reinterpret_cast<vec_t const*>(xp + static_cast<int64_t>(u) * x_l);
            }
#pragma unroll
            for (int u = 0; u < kU; ++u)
            {
                input_t const* vv = reinterpret_cast<input_t const*>(&v[u]);
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    xw[kW - 1 + u][e] = float(vv[e]);
                }
            }
#pragma unroll
            for (int u = 0; u < kU; ++u)
            {
                float acc[kE];
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    float a = bias_v[e];
#pragma unroll
                    for (int w = 0; w < kW; ++w)
                    {
                        a = __fmaf_rn(wt[w][e], xw[u + w][e], a);
                    }
                    acc[e] = a;
                }
                causal_conv1d_silu_vec<kSilu, kE>(acc, warpMask);
                input_t o_st[kE];
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    o_st[e] = input_t(acc[e]);
                }
                *reinterpret_cast<vec_t*>(op + static_cast<int64_t>(u) * o_l)
                    = *reinterpret_cast<vec_t const*>(&o_st[0]);
            }
#pragma unroll
            for (int w = 0; w < kW - 1; ++w)
            {
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    xw[w][e] = xw[kU + w][e];
                }
            }
            xp += static_cast<int64_t>(kU) * x_l;
            op += static_cast<int64_t>(kU) * o_l;
        }
        for (; t < seg; ++t)
        {
            vec_t cur = *reinterpret_cast<vec_t const*>(xp);
            input_t const* vv = reinterpret_cast<input_t const*>(&cur);
            float acc[kE];
#pragma unroll
            for (int e = 0; e < kE; ++e)
            {
                xw[kW - 1][e] = float(vv[e]);
                float a = bias_v[e];
#pragma unroll
                for (int w = 0; w < kW; ++w)
                {
                    a = __fmaf_rn(wt[w][e], xw[w][e], a);
                }
                acc[e] = a;
            }
            causal_conv1d_silu_vec<kSilu, kE>(acc, warpMask);
            input_t o_st[kE];
#pragma unroll
            for (int e = 0; e < kE; ++e)
            {
                o_st[e] = input_t(acc[e]);
            }
            *reinterpret_cast<vec_t*>(op) = *reinterpret_cast<vec_t const*>(&o_st[0]);
#pragma unroll
            for (int w = 0; w < kW - 1; ++w)
            {
#pragma unroll
                for (int e = 0; e < kE; ++e)
                {
                    xw[w][e] = xw[w + 1][e];
                }
            }
            xp += x_l;
            op += o_l;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch / dispatch for the channel-last forward kernel.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename weight_t, int kNThreads, int kWidth, int kEltsPerThread, int kChunkSizeL,
    int kUnroll, int kSiluMode>
void causal_conv1d_channellast_launch_one(ConvParamsBase& params, dim3 grid, bool hasCacheIndices, cudaStream_t stream)
{
    using Ktraits = Causal_conv1d_channellast_fwd_kernel_traits<input_t, weight_t, kNThreads, kWidth, kEltsPerThread,
        kChunkSizeL, kUnroll, kSiluMode>;
    if (hasCacheIndices)
    {
        causal_conv1d_channellast_fwd_kernel<Ktraits, true><<<grid, kNThreads, 0, stream>>>(params);
    }
    else
    {
        causal_conv1d_channellast_fwd_kernel<Ktraits, false><<<grid, kNThreads, 0, stream>>>(params);
    }
}

template <typename input_t, typename weight_t, int kNThreads, int kWidth, int kEltsPerThread, int kChunkSizeL,
    int kUnroll>
void causal_conv1d_channellast_launch(ConvParamsBase& params, int siluMode, cudaStream_t stream)
{
    constexpr int kChannelsPerBlock = kNThreads * kEltsPerThread;
    int const nC = (params.dim + kChannelsPerBlock - 1) / kChannelsPerBlock;
    // For varlen, params.seqlen is the *total* token count, which is an upper bound on any single
    // sequence length.  Blocks whose chunk lies past the end of their own sequence exit before
    // touching memory, so a batch of short sequences only pays block-scheduling cost.
    int const nL = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
    dim3 grid(nC, nL, params.batch);
    bool const hasCacheIndices = params.cache_indices_ptr != nullptr;

    switch (siluMode)
    {
    case 0:
        causal_conv1d_channellast_launch_one<input_t, weight_t, kNThreads, kWidth, kEltsPerThread, kChunkSizeL, kUnroll,
            0>(params, grid, hasCacheIndices, stream);
        break;
    case 1:
        causal_conv1d_channellast_launch_one<input_t, weight_t, kNThreads, kWidth, kEltsPerThread, kChunkSizeL, kUnroll,
            1>(params, grid, hasCacheIndices, stream);
        break;
    case 3:
        causal_conv1d_channellast_launch_one<input_t, weight_t, kNThreads, kWidth, kEltsPerThread, kChunkSizeL, kUnroll,
            3>(params, grid, hasCacheIndices, stream);
        break;
    default:
        causal_conv1d_channellast_launch_one<input_t, weight_t, kNThreads, kWidth, kEltsPerThread, kChunkSizeL, kUnroll,
            2>(params, grid, hasCacheIndices, stream);
        break;
    }
    TLLM_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t, typename weight_t, int kWidth>
void causal_conv1d_channellast_fwd_dispatch(ConvParamsBase& params, int siluMode, cudaStream_t stream)
{
    constexpr int kNThreads = 128;
    constexpr int kUnroll = 4;
    constexpr int kVecElts = 16 / sizeof(input_t);

    // 16B vector accesses are only legal when every address the kernel forms stays aligned.
    auto const xAddr = reinterpret_cast<uintptr_t>(params.x_ptr);
    auto const oAddr = reinterpret_cast<uintptr_t>(params.out_ptr);
    bool const vectorized = (params.dim % kVecElts == 0) && (xAddr % 16 == 0) && (oAddr % 16 == 0)
        && (params.x_l_stride % kVecElts == 0) && (params.out_l_stride % kVecElts == 0)
        && (params.x_batch_stride % kVecElts == 0) && (params.out_batch_stride % kVecElts == 0);

    if (!vectorized)
    {
        // Rare fallback (ragged channel count or an unaligned slice): same kernel, scalar accesses.
        causal_conv1d_channellast_launch<input_t, weight_t, kNThreads, kWidth, 1, 32, kUnroll>(
            params, siluMode, stream);
        return;
    }

    // Longer token chunks amortise the per-block setup, but only pay off once there is enough
    // work to fill the GPU several times over; otherwise prefer more, shorter chunks.
    int const nC = (params.dim + kNThreads * kVecElts - 1) / (kNThreads * kVecElts);
    int64_t const bigChunkBlocks
        = static_cast<int64_t>(nC) * ((params.seqlen + 63) / 64) * (params.batch > 0 ? params.batch : 1);
    if (bigChunkBlocks >= 1024)
    {
        causal_conv1d_channellast_launch<input_t, weight_t, kNThreads, kWidth, kVecElts, 64, kUnroll>(
            params, siluMode, stream);
    }
    else
    {
        causal_conv1d_channellast_launch<input_t, weight_t, kNThreads, kWidth, kVecElts, 32, kUnroll>(
            params, siluMode, stream);
    }
}

template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda_impl(ConvParamsBase& params, int siluMode, cudaStream_t stream)
{
    if (params.width == 2)
    {
        causal_conv1d_channellast_fwd_dispatch<input_t, weight_t, 2>(params, siluMode, stream);
    }
    else if (params.width == 3)
    {
        causal_conv1d_channellast_fwd_dispatch<input_t, weight_t, 3>(params, siluMode, stream);
    }
    else if (params.width == 4)
    {
        causal_conv1d_channellast_fwd_dispatch<input_t, weight_t, 4>(params, siluMode, stream);
    }
}

//! Chooses the SiLU implementation.
//!
//! Defaults to mode 1, which is bit-identical to causal_conv1d_fwd_kernel, so switching a
//! model between the channel-major and channel-last kernels does not move its outputs at
//! all.  That matters beyond accuracy: tests that assert batched and separate generation
//! produce identical greedy tokens sit on near-ties that any per-element rounding change
//! can flip.  Modes 2 (rcp.approx) and 3 (tanh + deep-negative vote, within one ulp of the
//! rounded 16-bit result) are faster and selectable via TRTLLM_CAUSAL_CONV1D_SILU_MODE.
inline int getCausalConv1dSiluMode(int defaultMode)
{
    static int const requested = []
    {
        auto const v = common::getIntEnv("TRTLLM_CAUSAL_CONV1D_SILU_MODE");
        int const m = v.has_value() ? v.value() : 0;
        return (m == 1 || m == 2 || m == 3) ? m : 0;
    }();
    return requested != 0 ? requested : defaultMode;
}

template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase& params, cudaStream_t stream)
{
    int siluMode = 0;
    if (params.silu_activation)
    {
        constexpr bool kIs16Bit = sizeof(input_t) == 2;
        siluMode = getCausalConv1dSiluMode(1);
        // The tanh form's accuracy argument is stated in terms of the 16-bit output grid; do not
        // use it when the result stays in fp32.
        if (siluMode == 3 && !kIs16Bit)
        {
            siluMode = 1;
        }
    }
    causal_conv1d_channellast_fwd_cuda_impl<input_t, weight_t>(params, siluMode, stream);
}

template void causal_conv1d_channellast_fwd_cuda<float, float>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<half, half>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<nv_bfloat16, nv_bfloat16>(ConvParamsBase& params, cudaStream_t stream);

template <int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits
{
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
};

template <typename Ktraits, bool kIsCircularBuffer, bool kHasConvStateIndices, bool kSiluActivation>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_update_kernel(ConvParamsBase params)
{
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    int const tidx = threadIdx.x;
    int const batch_id = blockIdx.x;
    int const channel_id = blockIdx.y * kNThreads + tidx;
    if (channel_id >= params.dim)
        return;

    input_t* x
        = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride + channel_id * params.x_c_stride;

    int conv_state_batch_coord;
    if constexpr (kHasConvStateIndices)
    {
        conv_state_batch_coord = params.conv_state_indices_ptr[batch_id];
        if (conv_state_batch_coord == params.pad_slot_id)
        {
            return;
        }
    }
    else
    {
        conv_state_batch_coord = batch_id;
    }
    input_t* conv_state = reinterpret_cast<input_t*>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride + channel_id * params.conv_state_c_stride;

    weight_t* weight = reinterpret_cast<weight_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t*>(params.bias_ptr)[channel_id]);

    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    float weight_vals[kWidth] = {0};
#pragma unroll
    for (int i = 0; i < kWidth; ++i)
    {
        weight_vals[i] = float(weight[i * params.weight_width_stride]);
    }

    float x_vals[kWidth] = {0};
    if constexpr (!kIsCircularBuffer)
    {
#pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i)
        {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }
#pragma unroll
        for (int i = 0; i < kWidth - 1; ++i)
        {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0)
            {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kWidth - 1;
             ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1)
        {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
        }
    }
#pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i)
    {
        input_t x_val = x[i * params.x_l_stride];
        if constexpr (!kIsCircularBuffer)
        {
            if (i < advance_len && state_len - advance_len + i >= 0)
            {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        }
        else
        {
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }
        x_vals[kWidth - 1] = float(x_val);
        float out_val = bias_val;
#pragma unroll
        for (int j = 0; j < kWidth; ++j)
        {
            out_val += weight_vals[j] * x_vals[j];
        }
        if constexpr (kSiluActivation)
        {
            out_val = out_val / (1 + expf(-out_val));
        }
        out[i * params.out_l_stride] = input_t(out_val);
// Shift the input buffer by 1
#pragma unroll
        for (int i = 0; i < kWidth - 1; ++i)
        {
            x_vals[i] = x_vals[i + 1];
        }
    }
}

// Specialized kernel for the dominant decode case (seqlen=1, non-circular, silu).
// Drops the per-token loop and circular-buffer bookkeeping from the general kernel.
template <typename Ktraits, bool kHasConvStateIndices>
__global__ __launch_bounds__(Ktraits::kNThreads) void causal_conv1d_update_kernel_sl1(ConvParamsBase params)
{
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    int const tidx = threadIdx.x;
    int const batch_id = blockIdx.x;
    int const channel_id = blockIdx.y * kNThreads + tidx;
    if (channel_id >= params.dim)
        return;

    int conv_state_batch_coord;
    if constexpr (kHasConvStateIndices)
    {
        conv_state_batch_coord = params.conv_state_indices_ptr[batch_id];
        if (conv_state_batch_coord == params.pad_slot_id)
            return;
    }
    else
    {
        conv_state_batch_coord = batch_id;
    }

    input_t* conv_state = reinterpret_cast<input_t*>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride + channel_id * params.conv_state_c_stride;
    weight_t* weight = reinterpret_cast<weight_t*>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t* x
        = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride + channel_id * params.x_c_stride;

    float w[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth; ++i)
        w[i] = float(__ldg(&weight[i * params.weight_width_stride]));

    float s[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i)
        s[i] = float(conv_state[i * params.conv_state_l_stride]);
    s[kWidth - 1] = float(x[0]);

    float out_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t*>(params.bias_ptr)[channel_id]);
#pragma unroll
    for (int i = 0; i < kWidth; ++i)
        out_val = __fmaf_rn(w[i], s[i], out_val);
    out_val = out_val * __frcp_rn(1.0f + __expf(-out_val));
    x[0] = input_t(out_val);

    // Shift conv_state left by one and append the new token.
#pragma unroll
    for (int i = 0; i < kWidth - 1; ++i)
        conv_state[i * params.conv_state_l_stride] = input_t(s[i + 1]);
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBase& params, cudaStream_t stream)
{
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    bool const hasConvStateIndices = params.conv_state_indices_ptr != nullptr;
    bool const isCircularBuffer = params.cache_seqlens != nullptr;

    // Fast path for the standard decode case (seqlen=1, non-circular, silu) when
    // conv_state holds exactly width-1 elements (no extra trailing padding to shift).
    if (params.seqlen == 1 && !isCircularBuffer && params.silu_activation && params.conv_state_len == params.width - 1)
    {
        BOOL_SWITCH(hasConvStateIndices, kHasCSI,
            [&]
            {
                auto kernel = &causal_conv1d_update_kernel_sl1<Ktraits, kHasCSI>;
                kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
            });
    }
    else
    {
        BOOL_SWITCH(isCircularBuffer, kIsCircBuf,
            [&]
            {
                BOOL_SWITCH(hasConvStateIndices, kHasCSI,
                    [&]
                    {
                        BOOL_SWITCH(params.silu_activation, kSilu,
                            [&]
                            {
                                auto kernel = &causal_conv1d_update_kernel<Ktraits, kIsCircBuf, kHasCSI, kSilu>;
                                kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
                            });
                    });
            });
    }
    TLLM_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase& params, cudaStream_t stream)
{
    // Wider blocks (128 vs 64 threads) halve block count, reducing scheduling overhead.
    constexpr int kNThreads = 128;
    if (params.width == 2)
    {
        causal_conv1d_update_launch<kNThreads, 2, input_t, weight_t>(params, stream);
    }
    else if (params.width == 3)
    {
        causal_conv1d_update_launch<kNThreads, 3, input_t, weight_t>(params, stream);
    }
    else if (params.width == 4)
    {
        causal_conv1d_update_launch<kNThreads, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_update_cuda<float, float>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_update_cuda<half, half>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_update_cuda<nv_bfloat16, nv_bfloat16>(ConvParamsBase& params, cudaStream_t stream);

} // namespace kernels::causal_conv1d

TRTLLM_NAMESPACE_END
