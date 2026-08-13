
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
