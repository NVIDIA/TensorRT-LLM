
/*
 * Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu
 * and https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu
 * Copyright (c) 2024, Tri Dao.
 *
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "tensorrt_llm/kernels/causalConv1d/causalConv1d.h"

namespace tensorrt_llm::kernels::causal_conv1d
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

template <typename Ktraits>
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

    int* cache_indices
        = params.cache_indices_ptr == nullptr ? nullptr : reinterpret_cast<int*>(params.cache_indices_ptr);
    int cache_index = cache_indices == nullptr ? batch_id : cache_indices[batch_id];
    // cache_index == params.pad_slot_id is defined as padding, so we exit early
    if (cache_index == params.pad_slot_id)
    {
        return;
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

    float weight_vals[kWidth];
#pragma unroll
    for (int i = 0; i < kWidth; ++i)
    {
        weight_vals[i] = float(weight[i * params.weight_width_stride]);
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
        __syncthreads();
        // Thread kNThreads - 1 don't write yet, so that thread 0 can read
        // the last elements of the previous chunk.
        if (tidx < kNThreads - 1)
        {
            smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
        }
        __syncthreads();
        reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        __syncthreads();
        // Now thread kNThreads - 1 can write the last elements of the current chunk.
        if (tidx == kNThreads - 1)
        {
            smem_exchange[tidx] = reinterpret_cast<vec_t*>(x_vals_load)[1];
        }

        float x_vals[2 * kNElts];
#pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i)
        {
            x_vals[i] = float(x_vals_load[i]);
        }

        float out_vals[kNElts];
#pragma unroll
        for (int i = 0; i < kNElts; ++i)
        {
            out_vals[i] = bias_val;
#pragma unroll
            for (int w = 0; w < kWidth; ++w)
            {
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
            }
        }

        if (params.silu_activation)
        {
#pragma unroll
            for (int i = 0; i < kNElts; ++i)
            {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
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

        int final_state_position = ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize);
        // in case the final state is separated between the last "smem_exchange" and
        // and the one before it (chunk = n_chunks - 1 and chunk = n_chunks - 2),
        // (which occurs when `final_state_position` is a non-positive index)
        // we load the correct data from smem_exchange from both chunks, the last chunk iteration and the one before it
        if (conv_states != nullptr && final_state_position < 0 && seqlen > kWidth)
        {
            input_t vals_load[kNElts] = {0};
            if ((chunk == n_chunks - 2) && (tidx == kNThreads - 1))
            {
                // chunk = n_chunks - 2, a segment of the final state sits in the last index
                reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[kNThreads - 1];
#pragma unroll
                for (int w = 0; w < -final_state_position; ++w)
                {
                    conv_states[w] = vals_load[kNElts + final_state_position + w];
                }
            }
            if ((chunk == n_chunks - 1) && tidx == 0)
            {
                // chunk = n_chunks - 1, the second segment of the final state first positions
                reinterpret_cast<vec_t*>(vals_load)[0] = smem_exchange[0];
                for (int w = -final_state_position; w < kWidth - 1; ++w)
                {
                    conv_states[w] = vals_load[w + final_state_position];
                }
                return;
            }
        }
    }
    // Final state is stored in the smem_exchange last token slot,
    // in case seqlen < kWidth, we would need to take the final state from the
    // initial state which is stored in conv_states
    // in case seqlen > kWidth, we would need to load the last kWidth - 1 data
    // and load it into conv_state accordingly
    int last_thread = ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize) / kNElts;
    if (conv_states != nullptr && tidx == last_thread)
    {
        input_t x_vals_load[kNElts * 2] = {0};
        // in case we are on the first kWidth tokens
        if (last_thread == 0 && seqlen < kWidth)
        {
            // Need to take the initial state
            reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[0];
            int const offset = seqlen - (kWidth - 1);
#pragma unroll
            for (int w = 0; w < kWidth - 1; ++w)
            {
                // pad the existing state
                if ((w - seqlen) >= 0 && has_initial_state)
                {
                    conv_states[w - seqlen] = conv_states[w];
                }
                else if ((w - seqlen) >= 0 && !has_initial_state)
                {
                    conv_states[w - seqlen] = input_t(0.0f);
                }
            }
#pragma unroll
            for (int w = 0; w < kWidth - 1; ++w)
            {
                if (offset + w >= 0)
                    conv_states[w] = x_vals_load[offset + w];
            }
        }
        else
        {
            // in case the final state is in between the threads data
            int const offset = ((seqlen - (kWidth - 1)) % (kNElts));
            if ((offset + kWidth - 2) >= kNElts && (last_thread + 1 < kNThreads))
            {
                // In case last_thread == kNThreads - 1, accessing last_thread + 1 will result in a
                // illegal access error on H100.
                // Therefore, we access last_thread + 1, only if the final state data sits there
                reinterpret_cast<vec_t*>(x_vals_load)[1] = smem_exchange[last_thread + 1];
            }
            reinterpret_cast<vec_t*>(x_vals_load)[0] = smem_exchange[last_thread];
#pragma unroll
            for (int w = 0; w < kWidth - 1; ++w)
            {
                conv_states[w] = x_vals_load[offset + w];
            }
        }
    }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase& params, cudaStream_t stream)
{
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    bool const kVarlen = params.query_start_loc_ptr != nullptr;
    BOOL_SWITCH(params.seqlen % kNElts == 0 && !kVarlen, kIsVecLoad,
        [&]
        {
            using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize;
            dim3 grid(params.batch, params.dim);

            auto kernel = &causal_conv1d_fwd_kernel<Ktraits>;

            if (kSmemSize >= 48 * 1024)
            {
                TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            }
            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            TLLM_CUDA_KERNEL_LAUNCH_CHECK();
        });
}

template <typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase& params, cudaStream_t stream)
{
    if (params.width == 2)
    {
        causal_conv1d_fwd_launch<128, 2, input_t, weight_t>(params, stream);
    }
    else if (params.width == 3)
    {
        causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream);
    }
    else if (params.width == 4)
    {
        causal_conv1d_fwd_launch<128, 4, input_t, weight_t>(params, stream);
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

template <typename Ktraits, bool kIsCircularBuffer>
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

    // If params.conv_state_batch_indices is set, then the conv state is gathered from the conv state tensor
    // along the batch axis. Otherwise, the conv state coordinate is the same as the batch id.
    int const conv_state_batch_coord
        = params.conv_state_indices_ptr == nullptr ? batch_id : params.conv_state_indices_ptr[batch_id];
    // conv_state_batch_coord == params.pad_slot_id is defined as padding so we exit early
    if (conv_state_batch_coord == params.pad_slot_id)
    {
        return;
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
        if (params.silu_activation)
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

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBase& params, cudaStream_t stream)
{
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    auto kernel = params.cache_seqlens == nullptr ? &causal_conv1d_update_kernel<Ktraits, false>
                                                  : &causal_conv1d_update_kernel<Ktraits, true>;
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    TLLM_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase& params, cudaStream_t stream)
{
    if (params.width == 2)
    {
        causal_conv1d_update_launch<64, 2, input_t, weight_t>(params, stream);
    }
    else if (params.width == 3)
    {
        causal_conv1d_update_launch<64, 3, input_t, weight_t>(params, stream);
    }
    else if (params.width == 4)
    {
        causal_conv1d_update_launch<64, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_update_cuda<float, float>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_update_cuda<half, half>(ConvParamsBase& params, cudaStream_t stream);
template void causal_conv1d_update_cuda<nv_bfloat16, nv_bfloat16>(ConvParamsBase& params, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::causal_conv1d
