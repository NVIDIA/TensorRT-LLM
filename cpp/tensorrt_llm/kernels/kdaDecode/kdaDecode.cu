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

#include "tensorrt_llm/kernels/kdaDecode/kdaDecode.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kdaDecode
{

namespace
{

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;

template <typename Index>
__device__ __forceinline__ float bf16_load(__nv_bfloat16 const* ptr, Index idx)
{
    return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float value)
{
    return __float2bfloat16(value);
}

template <bool kUseCacheGlobalStore>
__device__ __forceinline__ void store_state_float4(float* ptr, float4 value)
{
    if constexpr (kUseCacheGlobalStore)
    {
        __stcg(reinterpret_cast<float4*>(ptr), value);
    }
    else
    {
        *reinterpret_cast<float4*>(ptr) = value;
    }
}

__device__ __forceinline__ float sigmoid_fast(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x)
{
    return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x)
{
    return x > 20.0f ? x : log1pf(__expf(x));
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

// cp.async requires sm_80+; pre-SM80 falls back to a synchronous copy, so the
// commit/wait helpers become no-ops there.
__device__ __forceinline__ void cp_async_cg_16b(float* smem_ptr, float const* gmem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
#else
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<float4 const*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_group_1()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_group_2()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 2;\n" ::);
#endif
}

template <int kStageChunkV>
__device__ __forceinline__ void cp_async_state_chunk_stage(
    float* s_state, float const* state, int slot, int i_hv, int64_t state_slot_stride, int chunk, int stage)
{
    constexpr int kFloat4PerChunk = kStageChunkV * kDimK / 4;
    int const tid = threadIdx.x;
    int const v_base = chunk * kStageChunkV;
    for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kThreads)
    {
        int const elem = linear4 * 4;
        int const row = elem / kDimK;
        int const k = elem - row * kDimK;
        float* dst = s_state + (stage * kStageChunkV + row) * kDimK + k;
        int64_t const state_offset = static_cast<int64_t>(slot) * state_slot_stride
            + (static_cast<int64_t>(i_hv) * kDimV + v_base + row) * kDimK + k;
        float const* src = state + state_offset;
        cp_async_cg_16b(dst, src);
    }
    cp_async_commit();
}

template <int kCopyThreads>
__device__ __forceinline__ void cp_async_state_chunk_for(
    float* s_state, float const* state, int slot, int i_hv, int64_t state_slot_stride, int chunk)
{
    constexpr int kFloat4PerChunk = kChunkV * kDimK / 4;
    int const tid = threadIdx.x;
    int const stage = chunk & 1;
    int const v_base = chunk * kChunkV;
    for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kCopyThreads)
    {
        int const elem = linear4 * 4;
        int const row = elem / kDimK;
        int const k = elem - row * kDimK;
        float* dst = s_state + (stage * kChunkV + row) * kDimK + k;
        int64_t const state_offset = static_cast<int64_t>(slot) * state_slot_stride
            + (static_cast<int64_t>(i_hv) * kDimV + v_base + row) * kDimK + k;
        float const* src = state + state_offset;
        cp_async_cg_16b(dst, src);
    }
    cp_async_commit();
}

__device__ __forceinline__ void cp_async_state_chunk(
    float* s_state, float const* state, int slot, int i_hv, int64_t state_slot_stride, int chunk)
{
    cp_async_state_chunk_for<kThreads>(s_state, state, slot, i_hv, state_slot_stride, chunk);
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;

    float warp_total = warp_reduce_sum(value);
    if (lane == 0)
    {
        scratch[warp] = warp_total;
    }
    __syncthreads();

    float block_total = 0.0f;
    if (warp == 0)
    {
        block_total = lane < kWarps ? scratch[lane] : 0.0f;
        block_total = warp_reduce_sum(block_total);
        if (lane == 0)
        {
            scratch[0] = block_total;
        }
    }
    __syncthreads();
    return scratch[0];
}

struct Sum2
{
    float x;
    float y;
};

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

struct Sum4
{
    float a;
    float b;
    float c;
    float d;
};

__device__ __forceinline__ Sum4 warp_reduce_sum4(float a, float b, float c, float d)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_xor_sync(0xffffffffu, a, offset);
        b += __shfl_xor_sync(0xffffffffu, b, offset);
        c += __shfl_xor_sync(0xffffffffu, c, offset);
        d += __shfl_xor_sync(0xffffffffu, d, offset);
    }
    return {a, b, c, d};
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_for(float x, float y, float* scratch)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;

    float const warp_x = warp_reduce_sum(x);
    float const warp_y = warp_reduce_sum(y);
    if (lane == 0)
    {
        scratch[warp] = warp_x;
        scratch[kReduceWarps + warp] = warp_y;
    }
    __syncthreads();

    float block_x = 0.0f;
    float block_y = 0.0f;
    if (warp == 0)
    {
        block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
        block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
        block_x = warp_reduce_sum(block_x);
        block_y = warp_reduce_sum(block_y);
        if (lane == 0)
        {
            scratch[0] = block_x;
            scratch[1] = block_y;
        }
    }
    __syncthreads();
    return {scratch[0], scratch[1]};
}

__device__ __forceinline__ Sum2 block_reduce_sum2(float x, float y, float* scratch)
{
    return block_reduce_sum2_for<kWarps>(x, y, scratch);
}

template <int kReduceWarps>
__device__ __forceinline__ float block_reduce_sum_active_for(float value, float* scratch)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;

    float warp_total = 0.0f;
    if (warp < kReduceWarps)
    {
        warp_total = warp_reduce_sum(value);
    }
    if (lane == 0 && warp < kReduceWarps)
    {
        scratch[warp] = warp_total;
    }
    __syncthreads();

    float block_total = 0.0f;
    if (warp == 0)
    {
        block_total = lane < kReduceWarps ? scratch[lane] : 0.0f;
        block_total = warp_reduce_sum(block_total);
        if (lane == 0)
        {
            scratch[0] = block_total;
        }
    }
    __syncthreads();
    return scratch[0];
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_active_for(float x, float y, float* scratch)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;

    float warp_x = 0.0f;
    float warp_y = 0.0f;
    if (warp < kReduceWarps)
    {
        warp_x = warp_reduce_sum(x);
        warp_y = warp_reduce_sum(y);
    }
    if (lane == 0 && warp < kReduceWarps)
    {
        scratch[warp] = warp_x;
        scratch[kReduceWarps + warp] = warp_y;
    }
    __syncthreads();

    float block_x = 0.0f;
    float block_y = 0.0f;
    if (warp == 0)
    {
        block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
        block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
        block_x = warp_reduce_sum(block_x);
        block_y = warp_reduce_sum(block_y);
        if (lane == 0)
        {
            scratch[0] = block_x;
            scratch[1] = block_y;
        }
    }
    __syncthreads();
    return {scratch[0], scratch[1]};
}

template <bool kApplyOnorm, bool kAccumulateOnormSumsq = false, bool kUseStaticDecodeLayout = false,
    int kFixedHeads = 0, int kFixedValueHeads = 0, bool kUseHeadGrid = false, bool kUseCacheGlobalStore = false,
    bool kComputeOutputBeforeStore = false, bool kPreloadOnormParams = false,
    bool kIssueThirdStatePrefetchEarly = false, bool kUseActiveQkReduction = false, bool kUpdateConvState = false,
    bool kUseLowerBound = false, bool kApplyBetaSigmoid = true>
__global__ __launch_bounds__(kThreads, 2) void kda_decode_fusion_compact_heads_kernel(
    __nv_bfloat16 const* __restrict__ x_q, __nv_bfloat16 const* __restrict__ x_k, __nv_bfloat16 const* __restrict__ x_v,
    __nv_bfloat16 const* __restrict__ w_q_t, __nv_bfloat16 const* __restrict__ w_k_t,
    __nv_bfloat16 const* __restrict__ w_v_t, __nv_bfloat16 const* __restrict__ bias_q,
    __nv_bfloat16 const* __restrict__ bias_k, __nv_bfloat16 const* __restrict__ bias_v,
    __nv_bfloat16* __restrict__ cs_q, __nv_bfloat16* __restrict__ cs_k, __nv_bfloat16* __restrict__ cs_v,
    float const* __restrict__ a_log, __nv_bfloat16 const* __restrict__ g, float const* __restrict__ dt_bias,
    __nv_bfloat16 const* __restrict__ beta, __nv_bfloat16 const* __restrict__ onorm_g,
    float const* __restrict__ onorm_weight, int const* __restrict__ ssm_state_indices,
    int const* __restrict__ cu_seqlens, float* __restrict__ state, int64_t state_slot_stride,
    int64_t conv_state_slot_stride, __nv_bfloat16* __restrict__ out, int B, int H, int HV, float lower_bound,
    float scale, float onorm_eps, KdaDecodeIoLayout layout)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    int i_n;
    int i_hv;
    int i_h;
    int bos;
    int slot;
    if constexpr (kUseStaticDecodeLayout)
    {
        if constexpr (kUseHeadGrid)
        {
            i_n = blockIdx.x;
            i_hv = blockIdx.y;
        }
        else
        {
            int const nhv = blockIdx.x;
            i_n = nhv / kFixedValueHeads;
            i_hv = nhv - i_n * kFixedValueHeads;
        }
        i_h = i_hv;
        bos = i_n;
        slot = i_n;
    }
    else
    {
        int const nhv = blockIdx.x;
        i_n = nhv / HV;
        i_hv = nhv - i_n * HV;
        int const hv_per_h = HV / H;
        i_h = i_hv / hv_per_h;

        bos = cu_seqlens == nullptr ? i_n : cu_seqlens[i_n];
        int const eos = cu_seqlens == nullptr ? i_n + 1 : cu_seqlens[i_n + 1];
        if (eos <= bos)
        {
            return;
        }
        slot = ssm_state_indices[i_n];
    }

    int const hk_off = i_h * kDimK;
    int const hv_off = i_hv * kDimV;
    int const h_count = kUseStaticDecodeLayout ? kFixedHeads : H;
    int const hv_count = kUseStaticDecodeLayout ? kFixedValueHeads : HV;
    int const hkv_dim = h_count * kDimK;
    int const hvv_dim = hv_count * kDimV;
    int const conv_slot = kUpdateConvState ? slot : i_n;
    int64_t const output_head = static_cast<int64_t>(i_n) * hv_count + i_hv;

    constexpr int kStageChunkV = 32;
    constexpr int kStageCount = 3;
    constexpr int kStageNumChunks = kDimV / kStageChunkV;
    extern __shared__ float s_state[];
    __shared__ float s_q[kDimK];
    __shared__ float s_k[kDimK];
    __shared__ float s_decay[kDimK];
    __shared__ float s_v[kDimV];
    __shared__ float s_o[kDimV];
    __shared__ float s_reduce[kThreads];
    __shared__ float s_beta;
    float pre_onorm_gate = 0.0f;
    float pre_onorm_weight = 0.0f;

    cp_async_state_chunk_stage<kStageChunkV>(s_state, state, slot, i_hv, state_slot_stride, 0, 0);
    cp_async_state_chunk_stage<kStageChunkV>(s_state, state, slot, i_hv, state_slot_stride, 1, 1);

    if constexpr (kUpdateConvState)
    {
        if (tid < kDimK)
        {
            int const k = tid;
            int const hk = hk_off + k;
            int64_t const cs_base = static_cast<int64_t>(slot) * conv_state_slot_stride + hk * kConvStateWidth;
            float const exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

            float q_acc = bf16_load(bias_q, hk);
            float k_acc = bf16_load(bias_k, hk);
            __nv_bfloat16 q_shift0 = __float2bfloat16(0.0f);
            __nv_bfloat16 q_shift1 = __float2bfloat16(0.0f);
            __nv_bfloat16 k_shift0 = __float2bfloat16(0.0f);
            __nv_bfloat16 k_shift1 = __float2bfloat16(0.0f);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                const __nv_bfloat16 q_state = cs_q[cs_base + w];
                const __nv_bfloat16 k_state = cs_k[cs_base + w];
                q_acc += __bfloat162float(q_state) * bf16_load(w_q_t, w * hkv_dim + hk);
                k_acc += __bfloat162float(k_state) * bf16_load(w_k_t, w * hkv_dim + hk);
                if (w == 1)
                {
                    q_shift0 = q_state;
                    k_shift0 = k_state;
                }
                else if (w == 2)
                {
                    q_shift1 = q_state;
                    k_shift1 = k_state;
                }
            }
            const __nv_bfloat16 q_new = x_q[bos * layout.xQRowStride + i_h * kDimK + k];
            const __nv_bfloat16 k_new = x_k[bos * layout.xKRowStride + i_h * kDimK + k];
            q_acc += __bfloat162float(q_new) * bf16_load(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
            k_acc += __bfloat162float(k_new) * bf16_load(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

            cs_q[cs_base + 0] = q_shift0;
            cs_q[cs_base + 1] = q_shift1;
            cs_q[cs_base + 2] = q_new;
            cs_k[cs_base + 0] = k_shift0;
            cs_k[cs_base + 1] = k_shift1;
            cs_k[cs_base + 2] = k_new;

            s_q[k] = silu_fast(q_acc);
            s_k[k] = silu_fast(k_acc);

            float const g_raw = bf16_load(g, bos * layout.gateRowStride + i_hv * kDimK + k) + dt_bias[hk];
            if constexpr (kUseLowerBound)
            {
                s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
            }
            else
            {
                s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
            }
        }
    }
    else
    {
        if (tid < kDimK)
        {
            int const k = tid;
            int const hk = hk_off + k;
            float const exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

            float q_acc = bf16_load(bias_q, hk);
            float k_acc = bf16_load(bias_k, hk);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                int64_t const cs_idx = (static_cast<int64_t>(conv_slot) * hkv_dim + hk) * kConvStateWidth + w;
                q_acc += bf16_load(cs_q, cs_idx) * bf16_load(w_q_t, w * hkv_dim + hk);
                k_acc += bf16_load(cs_k, cs_idx) * bf16_load(w_k_t, w * hkv_dim + hk);
            }
            q_acc += bf16_load(x_q, bos * layout.xQRowStride + i_h * kDimK + k)
                * bf16_load(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
            k_acc += bf16_load(x_k, bos * layout.xKRowStride + i_h * kDimK + k)
                * bf16_load(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

            s_q[k] = silu_fast(q_acc);
            s_k[k] = silu_fast(k_acc);

            float const g_raw = bf16_load(g, bos * layout.gateRowStride + i_hv * kDimK + k) + dt_bias[hk];
            if constexpr (kUseLowerBound)
            {
                s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
            }
            else
            {
                s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
            }
        }
    }

    if constexpr (kUpdateConvState)
    {
        if (tid < kDimV)
        {
            int const v = tid;
            int const hvv = hv_off + v;
            int64_t const cs_base = static_cast<int64_t>(slot) * conv_state_slot_stride + hvv * kConvStateWidth;
            float v_acc = bf16_load(bias_v, hvv);
            __nv_bfloat16 v_shift0 = __float2bfloat16(0.0f);
            __nv_bfloat16 v_shift1 = __float2bfloat16(0.0f);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                const __nv_bfloat16 v_state = cs_v[cs_base + w];
                v_acc += __bfloat162float(v_state) * bf16_load(w_v_t, w * hvv_dim + hvv);
                if (w == 1)
                {
                    v_shift0 = v_state;
                }
                else if (w == 2)
                {
                    v_shift1 = v_state;
                }
            }
            const __nv_bfloat16 v_new = x_v[bos * layout.xVRowStride + i_hv * kDimV + v];
            v_acc += __bfloat162float(v_new) * bf16_load(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
            cs_v[cs_base + 0] = v_shift0;
            cs_v[cs_base + 1] = v_shift1;
            cs_v[cs_base + 2] = v_new;
            s_v[v] = silu_fast(v_acc);

            if constexpr (kApplyOnorm && kPreloadOnormParams)
            {
                pre_onorm_gate
                    = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + v));
                pre_onorm_weight = onorm_weight[v];
            }
        }
    }
    else
    {
        if (tid < kDimV)
        {
            int const v = tid;
            int const hvv = hv_off + v;

            float v_acc = bf16_load(bias_v, hvv);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                int64_t const cs_idx = (static_cast<int64_t>(conv_slot) * hvv_dim + hvv) * kConvStateWidth + w;
                v_acc += bf16_load(cs_v, cs_idx) * bf16_load(w_v_t, w * hvv_dim + hvv);
            }
            v_acc += bf16_load(x_v, bos * layout.xVRowStride + i_hv * kDimV + v)
                * bf16_load(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
            s_v[v] = silu_fast(v_acc);

            if constexpr (kApplyOnorm && kPreloadOnormParams)
            {
                pre_onorm_gate
                    = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + v));
                pre_onorm_weight = onorm_weight[v];
            }
        }
    }

    if (tid == 0)
    {
        float const beta_raw = bf16_load(beta, bos * layout.betaRowStride + i_hv);
        if constexpr (kApplyBetaSigmoid)
        {
            s_beta = sigmoid_fast(beta_raw);
        }
        else
        {
            s_beta = beta_raw;
        }
    }
    __syncthreads();

    if constexpr (kIssueThirdStatePrefetchEarly && kStageNumChunks > 2)
    {
        cp_async_state_chunk_stage<kStageChunkV>(s_state, state, slot, i_hv, state_slot_stride, 2, 2);
    }

    float const q_sq = tid < kDimK ? s_q[tid] * s_q[tid] : 0.0f;
    float const k_sq = tid < kDimK ? s_k[tid] * s_k[tid] : 0.0f;
    Sum2 qk_sum;
    if constexpr (kUseActiveQkReduction)
    {
        qk_sum = block_reduce_sum2_active_for<kDimK / 32>(q_sq, k_sq, s_reduce);
    }
    else
    {
        qk_sum = block_reduce_sum2(q_sq, k_sq, s_reduce);
    }
    if (tid < kDimK)
    {
        s_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
        s_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
    }
    __syncthreads();

    int const k_base = lane * 4;
    const float4 q4 = *reinterpret_cast<float4 const*>(s_q + k_base);
    const float4 k4 = *reinterpret_cast<float4 const*>(s_k + k_base);
    const float4 decay4 = *reinterpret_cast<float4 const*>(s_decay + k_base);
    float r_q[4] = {q4.x, q4.y, q4.z, q4.w};
    float r_k[4] = {k4.x, k4.y, k4.z, k4.w};
    float r_decay[4] = {decay4.x, decay4.y, decay4.z, decay4.w};
    float o_sumsq = 0.0f;

#pragma unroll
    for (int chunk = 0; chunk < kStageNumChunks; ++chunk)
    {
        if constexpr (kIssueThirdStatePrefetchEarly)
        {
            if (chunk == 0)
            {
                cp_async_wait_group_2();
            }
            else if (chunk + 1 < kStageNumChunks)
            {
                cp_async_wait_group_1();
            }
            else
            {
                cp_async_wait_group_0();
            }
        }
        else
        {
            if (chunk + 1 < kStageNumChunks)
            {
                cp_async_wait_group_1();
            }
            else
            {
                cp_async_wait_group_0();
            }
        }
        __syncwarp();

        int const prefetch = chunk + 2;
        if constexpr (kIssueThirdStatePrefetchEarly)
        {
            if (prefetch < kStageNumChunks && prefetch >= 3)
            {
                cp_async_state_chunk_stage<kStageChunkV>(
                    s_state, state, slot, i_hv, state_slot_stride, prefetch, prefetch % kStageCount);
            }
        }
        else
        {
            if (prefetch < kStageNumChunks)
            {
                cp_async_state_chunk_stage<kStageChunkV>(
                    s_state, state, slot, i_hv, state_slot_stride, prefetch, prefetch % kStageCount);
            }
        }

        int const v_row_a = warp;
        int const v_row_b = warp + kWarps;
        int const v_row_c = warp + 2 * kWarps;
        int const v_row_d = warp + 3 * kWarps;
        int const v0 = chunk * kChunkV + v_row_a;
        int const v1 = chunk * kChunkV + v_row_b;
        int const v2 = chunk * kChunkV + v_row_c;
        int const v3 = chunk * kChunkV + v_row_d;
        float h_a_vals[4];
        float h_b_vals[4];
        float h_c_vals[4];
        float h_d_vals[4];
        float dot_hk_a = 0.0f;
        float dot_hk_b = 0.0f;
        float dot_hk_c = 0.0f;
        float dot_hk_d = 0.0f;

        float const* state_stage = s_state + (chunk % kStageCount) * kStageChunkV * kDimK;
        const float4 raw_h_a = *reinterpret_cast<float4 const*>(state_stage + v_row_a * kDimK + k_base);
        const float4 raw_h_b = *reinterpret_cast<float4 const*>(state_stage + v_row_b * kDimK + k_base);
        const float4 raw_h_c = *reinterpret_cast<float4 const*>(state_stage + v_row_c * kDimK + k_base);
        const float4 raw_h_d = *reinterpret_cast<float4 const*>(state_stage + v_row_d * kDimK + k_base);
        h_a_vals[0] = raw_h_a.x * r_decay[0];
        h_a_vals[1] = raw_h_a.y * r_decay[1];
        h_a_vals[2] = raw_h_a.z * r_decay[2];
        h_a_vals[3] = raw_h_a.w * r_decay[3];
        h_b_vals[0] = raw_h_b.x * r_decay[0];
        h_b_vals[1] = raw_h_b.y * r_decay[1];
        h_b_vals[2] = raw_h_b.z * r_decay[2];
        h_b_vals[3] = raw_h_b.w * r_decay[3];
        h_c_vals[0] = raw_h_c.x * r_decay[0];
        h_c_vals[1] = raw_h_c.y * r_decay[1];
        h_c_vals[2] = raw_h_c.z * r_decay[2];
        h_c_vals[3] = raw_h_c.w * r_decay[3];
        h_d_vals[0] = raw_h_d.x * r_decay[0];
        h_d_vals[1] = raw_h_d.y * r_decay[1];
        h_d_vals[2] = raw_h_d.z * r_decay[2];
        h_d_vals[3] = raw_h_d.w * r_decay[3];
        dot_hk_a = h_a_vals[0] * r_k[0] + h_a_vals[1] * r_k[1] + h_a_vals[2] * r_k[2] + h_a_vals[3] * r_k[3];
        dot_hk_b = h_b_vals[0] * r_k[0] + h_b_vals[1] * r_k[1] + h_b_vals[2] * r_k[2] + h_b_vals[3] * r_k[3];
        dot_hk_c = h_c_vals[0] * r_k[0] + h_c_vals[1] * r_k[1] + h_c_vals[2] * r_k[2] + h_c_vals[3] * r_k[3];
        dot_hk_d = h_d_vals[0] * r_k[0] + h_d_vals[1] * r_k[1] + h_d_vals[2] * r_k[2] + h_d_vals[3] * r_k[3];

        const Sum4 dot_hk = warp_reduce_sum4(dot_hk_a, dot_hk_b, dot_hk_c, dot_hk_d);
        float const v_new0 = (s_v[v0] - dot_hk.a) * s_beta;
        float const v_new1 = (s_v[v1] - dot_hk.b) * s_beta;
        float const v_new2 = (s_v[v2] - dot_hk.c) * s_beta;
        float const v_new3 = (s_v[v3] - dot_hk.d) * s_beta;

        float dot_hq_a = 0.0f;
        float dot_hq_b = 0.0f;
        float dot_hq_c = 0.0f;
        float dot_hq_d = 0.0f;
        int64_t const state_head_offset
            = static_cast<int64_t>(slot) * state_slot_stride + static_cast<int64_t>(i_hv) * kDimV * kDimK;
        int64_t const state_idx_a = state_head_offset + static_cast<int64_t>(v0) * kDimK + k_base;
        int64_t const state_idx_b = state_head_offset + static_cast<int64_t>(v1) * kDimK + k_base;
        int64_t const state_idx_c = state_head_offset + static_cast<int64_t>(v2) * kDimK + k_base;
        int64_t const state_idx_d = state_head_offset + static_cast<int64_t>(v3) * kDimK + k_base;
        float const h_a_0 = h_a_vals[0] + r_k[0] * v_new0;
        float const h_a_1 = h_a_vals[1] + r_k[1] * v_new0;
        float const h_a_2 = h_a_vals[2] + r_k[2] * v_new0;
        float const h_a_3 = h_a_vals[3] + r_k[3] * v_new0;
        float const h_b_0 = h_b_vals[0] + r_k[0] * v_new1;
        float const h_b_1 = h_b_vals[1] + r_k[1] * v_new1;
        float const h_b_2 = h_b_vals[2] + r_k[2] * v_new1;
        float const h_b_3 = h_b_vals[3] + r_k[3] * v_new1;
        float const h_c_0 = h_c_vals[0] + r_k[0] * v_new2;
        float const h_c_1 = h_c_vals[1] + r_k[1] * v_new2;
        float const h_c_2 = h_c_vals[2] + r_k[2] * v_new2;
        float const h_c_3 = h_c_vals[3] + r_k[3] * v_new2;
        float const h_d_0 = h_d_vals[0] + r_k[0] * v_new3;
        float const h_d_1 = h_d_vals[1] + r_k[1] * v_new3;
        float const h_d_2 = h_d_vals[2] + r_k[2] * v_new3;
        float const h_d_3 = h_d_vals[3] + r_k[3] * v_new3;
        if constexpr (kComputeOutputBeforeStore)
        {
            dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
            dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
            dot_hq_c = h_c_0 * r_q[0] + h_c_1 * r_q[1] + h_c_2 * r_q[2] + h_c_3 * r_q[3];
            dot_hq_d = h_d_0 * r_q[0] + h_d_1 * r_q[1] + h_d_2 * r_q[2] + h_d_3 * r_q[3];
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_c, make_float4(h_c_0, h_c_1, h_c_2, h_c_3));
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_d, make_float4(h_d_0, h_d_1, h_d_2, h_d_3));
        }
        else
        {
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_c, make_float4(h_c_0, h_c_1, h_c_2, h_c_3));
            store_state_float4<kUseCacheGlobalStore>(state + state_idx_d, make_float4(h_d_0, h_d_1, h_d_2, h_d_3));
            dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
            dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
            dot_hq_c = h_c_0 * r_q[0] + h_c_1 * r_q[1] + h_c_2 * r_q[2] + h_c_3 * r_q[3];
            dot_hq_d = h_d_0 * r_q[0] + h_d_1 * r_q[1] + h_d_2 * r_q[2] + h_d_3 * r_q[3];
        }

        const Sum4 dot_hq = warp_reduce_sum4(dot_hq_a, dot_hq_b, dot_hq_c, dot_hq_d);
        if (lane == 0)
        {
            s_o[v0] = dot_hq.a;
            s_o[v1] = dot_hq.b;
            s_o[v2] = dot_hq.c;
            s_o[v3] = dot_hq.d;
            if constexpr (kApplyOnorm && kAccumulateOnormSumsq)
            {
                o_sumsq += dot_hq.a * dot_hq.a + dot_hq.b * dot_hq.b + dot_hq.c * dot_hq.c + dot_hq.d * dot_hq.d;
            }
        }
    }
    __syncthreads();

    if constexpr (kApplyOnorm)
    {
        if constexpr (kAccumulateOnormSumsq)
        {
            if (lane == 0)
            {
                s_reduce[warp] = o_sumsq;
            }
            __syncthreads();

            float total_sumsq = 0.0f;
            if (warp == 0)
            {
                total_sumsq = lane < kWarps ? s_reduce[lane] : 0.0f;
                total_sumsq = warp_reduce_sum(total_sumsq);
                if (lane == 0)
                {
                    s_reduce[0] = total_sumsq;
                }
            }
            __syncthreads();

            if (tid < kDimV)
            {
                int64_t const out_idx = output_head * kDimV + tid;
                float const raw_o = s_o[tid];
                float const rstd = rsqrtf(s_reduce[0] / static_cast<float>(kDimV) + onorm_eps);
                float gate;
                float weight;
                if constexpr (kPreloadOnormParams)
                {
                    gate = pre_onorm_gate;
                    weight = pre_onorm_weight;
                }
                else
                {
                    gate = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + tid));
                    weight = onorm_weight[tid];
                }
                float const y = raw_o * rstd * weight * gate;
                out[out_idx] = bf16_store(y);
            }
        }
        else
        {
            float const raw_o = tid < kDimV ? s_o[tid] : 0.0f;
            float const o_sq = raw_o * raw_o;
            float const sumsq = block_reduce_sum(o_sq, s_reduce);

            if (tid < kDimV)
            {
                int64_t const out_idx = output_head * kDimV + tid;
                float const rstd = rsqrtf(sumsq / static_cast<float>(kDimV) + onorm_eps);
                float gate;
                float weight;
                if constexpr (kPreloadOnormParams)
                {
                    gate = pre_onorm_gate;
                    weight = pre_onorm_weight;
                }
                else
                {
                    gate = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + tid));
                    weight = onorm_weight[tid];
                }
                float const y = raw_o * rstd * weight * gate;
                out[out_idx] = bf16_store(y);
            }
        }
    }
    else
    {
        if (tid < kDimV)
        {
            int64_t const out_idx = output_head * kDimV + tid;
            out[out_idx] = bf16_store(s_o[tid]);
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <bool kApplyOnorm, bool kUseStaticDecodeLayout = false, int kFixedHeads = 0, int kFixedValueHeads = 0,
    bool kUseHeadGrid = false, bool kAccumulateOnormSumsq = false, bool kUseActiveQkReduction = false,
    bool kUseCacheGlobalStore = false, bool kComputeOutputBeforeStore = false, bool kSkipWarpSync = false,
    bool kPreloadOnormParams = false, bool kPrefetchNextStateChunk = false, bool kUseActiveOnormReduction = false,
    bool kUpdateConvState = false, bool kUseLowerBound = false, bool kApplyBetaSigmoid = true>
__global__ __launch_bounds__(kThreads, 2) void kda_decode_fusion_many_heads_kernel(
    __nv_bfloat16 const* __restrict__ x_q, __nv_bfloat16 const* __restrict__ x_k, __nv_bfloat16 const* __restrict__ x_v,
    __nv_bfloat16 const* __restrict__ w_q_t, __nv_bfloat16 const* __restrict__ w_k_t,
    __nv_bfloat16 const* __restrict__ w_v_t, __nv_bfloat16 const* __restrict__ bias_q,
    __nv_bfloat16 const* __restrict__ bias_k, __nv_bfloat16 const* __restrict__ bias_v,
    __nv_bfloat16* __restrict__ cs_q, __nv_bfloat16* __restrict__ cs_k, __nv_bfloat16* __restrict__ cs_v,
    float const* __restrict__ a_log, __nv_bfloat16 const* __restrict__ g, float const* __restrict__ dt_bias,
    __nv_bfloat16 const* __restrict__ beta, __nv_bfloat16 const* __restrict__ onorm_g,
    float const* __restrict__ onorm_weight, int const* __restrict__ ssm_state_indices,
    int const* __restrict__ cu_seqlens, float* __restrict__ state, int64_t state_slot_stride,
    int64_t conv_state_slot_stride, __nv_bfloat16* __restrict__ out, int B, int H, int HV, float lower_bound,
    float scale, float onorm_eps, KdaDecodeIoLayout layout)
{
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    int i_n;
    int i_hv;
    int i_h;
    int bos;
    int slot;
    if constexpr (kUseStaticDecodeLayout)
    {
        if constexpr (kUseHeadGrid)
        {
            i_n = blockIdx.x;
            i_hv = blockIdx.y;
        }
        else
        {
            int const nhv = blockIdx.x;
            i_n = nhv / kFixedValueHeads;
            i_hv = nhv - i_n * kFixedValueHeads;
        }
        i_h = i_hv;
        bos = i_n;
        slot = i_n;
    }
    else
    {
        int const nhv = blockIdx.x;
        i_n = nhv / HV;
        i_hv = nhv - i_n * HV;
        int const hv_per_h = HV / H;
        i_h = i_hv / hv_per_h;

        bos = cu_seqlens == nullptr ? i_n : cu_seqlens[i_n];
        int const eos = cu_seqlens == nullptr ? i_n + 1 : cu_seqlens[i_n + 1];
        if (eos <= bos)
        {
            return;
        }
        slot = ssm_state_indices[i_n];
    }

    int const hk_off = i_h * kDimK;
    int const hv_off = i_hv * kDimV;
    int const h_count = kUseStaticDecodeLayout ? kFixedHeads : H;
    int const hv_count = kUseStaticDecodeLayout ? kFixedValueHeads : HV;
    int const hkv_dim = h_count * kDimK;
    int const hvv_dim = hv_count * kDimV;
    int const conv_slot = kUpdateConvState ? slot : i_n;
    int64_t const output_head = static_cast<int64_t>(i_n) * hv_count + i_hv;

    __shared__ float s_state[2][kChunkV][kDimK];
    __shared__ float s_q[kDimK];
    __shared__ float s_k[kDimK];
    __shared__ float s_decay[kDimK];
    __shared__ float s_v[kDimV];
    __shared__ float s_o[kDimV];
    __shared__ float s_reduce[kThreads];
    __shared__ float s_beta;
    float pre_onorm_gate = 0.0f;
    float pre_onorm_weight = 0.0f;

    cp_async_state_chunk(&s_state[0][0][0], state, slot, i_hv, state_slot_stride, 0);

    if constexpr (kUpdateConvState)
    {
        if (tid < kDimK)
        {
            int const k = tid;
            int const hk = hk_off + k;
            int64_t const cs_base = static_cast<int64_t>(slot) * conv_state_slot_stride + hk * kConvStateWidth;
            float const exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

            float q_acc = bf16_load(bias_q, hk);
            float k_acc = bf16_load(bias_k, hk);
            __nv_bfloat16 q_shift0 = __float2bfloat16(0.0f);
            __nv_bfloat16 q_shift1 = __float2bfloat16(0.0f);
            __nv_bfloat16 k_shift0 = __float2bfloat16(0.0f);
            __nv_bfloat16 k_shift1 = __float2bfloat16(0.0f);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                const __nv_bfloat16 q_state = cs_q[cs_base + w];
                const __nv_bfloat16 k_state = cs_k[cs_base + w];
                q_acc += __bfloat162float(q_state) * bf16_load(w_q_t, w * hkv_dim + hk);
                k_acc += __bfloat162float(k_state) * bf16_load(w_k_t, w * hkv_dim + hk);
                if (w == 1)
                {
                    q_shift0 = q_state;
                    k_shift0 = k_state;
                }
                else if (w == 2)
                {
                    q_shift1 = q_state;
                    k_shift1 = k_state;
                }
            }
            const __nv_bfloat16 q_new = x_q[bos * layout.xQRowStride + i_h * kDimK + k];
            const __nv_bfloat16 k_new = x_k[bos * layout.xKRowStride + i_h * kDimK + k];
            q_acc += __bfloat162float(q_new) * bf16_load(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
            k_acc += __bfloat162float(k_new) * bf16_load(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

            cs_q[cs_base + 0] = q_shift0;
            cs_q[cs_base + 1] = q_shift1;
            cs_q[cs_base + 2] = q_new;
            cs_k[cs_base + 0] = k_shift0;
            cs_k[cs_base + 1] = k_shift1;
            cs_k[cs_base + 2] = k_new;

            s_q[k] = silu_fast(q_acc);
            s_k[k] = silu_fast(k_acc);

            float const g_raw = bf16_load(g, bos * layout.gateRowStride + i_hv * kDimK + k) + dt_bias[hk];
            if constexpr (kUseLowerBound)
            {
                s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
            }
            else
            {
                s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
            }
        }
    }
    else
    {
        if (tid < kDimK)
        {
            int const k = tid;
            int const hk = hk_off + k;
            float const exp_a = __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

            float q_acc = bf16_load(bias_q, hk);
            float k_acc = bf16_load(bias_k, hk);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                int64_t const cs_idx = (static_cast<int64_t>(conv_slot) * hkv_dim + hk) * kConvStateWidth + w;
                q_acc += bf16_load(cs_q, cs_idx) * bf16_load(w_q_t, w * hkv_dim + hk);
                k_acc += bf16_load(cs_k, cs_idx) * bf16_load(w_k_t, w * hkv_dim + hk);
            }
            q_acc += bf16_load(x_q, bos * layout.xQRowStride + i_h * kDimK + k)
                * bf16_load(w_q_t, (kKernelWidth - 1) * hkv_dim + hk);
            k_acc += bf16_load(x_k, bos * layout.xKRowStride + i_h * kDimK + k)
                * bf16_load(w_k_t, (kKernelWidth - 1) * hkv_dim + hk);

            s_q[k] = silu_fast(q_acc);
            s_k[k] = silu_fast(k_acc);

            float const g_raw = bf16_load(g, bos * layout.gateRowStride + i_hv * kDimK + k) + dt_bias[hk];
            if constexpr (kUseLowerBound)
            {
                s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
            }
            else
            {
                s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
            }
        }
    }

    if constexpr (kUpdateConvState)
    {
        if (tid < kDimV)
        {
            int const v = tid;
            int const hvv = hv_off + v;
            int64_t const cs_base = static_cast<int64_t>(slot) * conv_state_slot_stride + hvv * kConvStateWidth;
            float v_acc = bf16_load(bias_v, hvv);
            __nv_bfloat16 v_shift0 = __float2bfloat16(0.0f);
            __nv_bfloat16 v_shift1 = __float2bfloat16(0.0f);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                const __nv_bfloat16 v_state = cs_v[cs_base + w];
                v_acc += __bfloat162float(v_state) * bf16_load(w_v_t, w * hvv_dim + hvv);
                if (w == 1)
                {
                    v_shift0 = v_state;
                }
                else if (w == 2)
                {
                    v_shift1 = v_state;
                }
            }
            const __nv_bfloat16 v_new = x_v[bos * layout.xVRowStride + i_hv * kDimV + v];
            v_acc += __bfloat162float(v_new) * bf16_load(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
            cs_v[cs_base + 0] = v_shift0;
            cs_v[cs_base + 1] = v_shift1;
            cs_v[cs_base + 2] = v_new;
            s_v[v] = silu_fast(v_acc);

            if constexpr (kApplyOnorm && kPreloadOnormParams)
            {
                pre_onorm_gate
                    = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + v));
                pre_onorm_weight = onorm_weight[v];
            }
        }
    }
    else
    {
        if (tid < kDimV)
        {
            int const v = tid;
            int const hvv = hv_off + v;

            float v_acc = bf16_load(bias_v, hvv);
#pragma unroll
            for (int w = 0; w < kConvStateWidth; ++w)
            {
                int64_t const cs_idx = (static_cast<int64_t>(conv_slot) * hvv_dim + hvv) * kConvStateWidth + w;
                v_acc += bf16_load(cs_v, cs_idx) * bf16_load(w_v_t, w * hvv_dim + hvv);
            }
            v_acc += bf16_load(x_v, bos * layout.xVRowStride + i_hv * kDimV + v)
                * bf16_load(w_v_t, (kKernelWidth - 1) * hvv_dim + hvv);
            s_v[v] = silu_fast(v_acc);

            if constexpr (kApplyOnorm && kPreloadOnormParams)
            {
                pre_onorm_gate
                    = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + v));
                pre_onorm_weight = onorm_weight[v];
            }
        }
    }

    if (tid == 0)
    {
        float const beta_raw = bf16_load(beta, bos * layout.betaRowStride + i_hv);
        if constexpr (kApplyBetaSigmoid)
        {
            s_beta = sigmoid_fast(beta_raw);
        }
        else
        {
            s_beta = beta_raw;
        }
    }
    __syncthreads();

    if constexpr (kPrefetchNextStateChunk && kNumChunks > 1)
    {
        cp_async_state_chunk(&s_state[0][0][0], state, slot, i_hv, state_slot_stride, 1);
    }

    float const q_sq = tid < kDimK ? s_q[tid] * s_q[tid] : 0.0f;
    float const k_sq = tid < kDimK ? s_k[tid] * s_k[tid] : 0.0f;
    Sum2 qk_sum;
    if constexpr (kUseActiveQkReduction)
    {
        qk_sum = block_reduce_sum2_active_for<kDimK / 32>(q_sq, k_sq, s_reduce);
    }
    else
    {
        qk_sum = block_reduce_sum2(q_sq, k_sq, s_reduce);
    }
    if (tid < kDimK)
    {
        s_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
        s_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
    }
    __syncthreads();

    int const k_base = lane * 4;
    const float4 q4 = *reinterpret_cast<float4 const*>(s_q + k_base);
    const float4 k4 = *reinterpret_cast<float4 const*>(s_k + k_base);
    const float4 decay4 = *reinterpret_cast<float4 const*>(s_decay + k_base);
    float r_q[4] = {q4.x, q4.y, q4.z, q4.w};
    float r_k[4] = {k4.x, k4.y, k4.z, k4.w};
    float r_decay[4] = {decay4.x, decay4.y, decay4.z, decay4.w};
    float o_sumsq = 0.0f;

#pragma unroll
    for (int chunk = 0; chunk < kNumChunks; ++chunk)
    {
        if constexpr (kPrefetchNextStateChunk && kNumChunks > 1)
        {
            if (chunk + 1 < kNumChunks)
            {
                cp_async_wait_group_1();
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
        if constexpr (!kSkipWarpSync)
        {
            __syncwarp();
        }

        if constexpr (!kPrefetchNextStateChunk)
        {
            if (chunk + 1 < kNumChunks)
            {
                cp_async_state_chunk(&s_state[0][0][0], state, slot, i_hv, state_slot_stride, chunk + 1);
            }
        }

#pragma unroll
        for (int row = 0; row < kRowsPerWarp; row += 2)
        {
            int const v_row_a = warp + row * kWarps;
            int const v_row_b = warp + (row + 1) * kWarps;
            int const v0 = chunk * kChunkV + v_row_a;
            int const v1 = chunk * kChunkV + v_row_b;
            float h_a_vals[4];
            float h_b_vals[4];
            float dot_hk_a = 0.0f;
            float dot_hk_b = 0.0f;

            const float4 raw_h_a = *reinterpret_cast<float4 const*>(&s_state[chunk & 1][v_row_a][k_base]);
            const float4 raw_h_b = *reinterpret_cast<float4 const*>(&s_state[chunk & 1][v_row_b][k_base]);
            h_a_vals[0] = raw_h_a.x * r_decay[0];
            h_a_vals[1] = raw_h_a.y * r_decay[1];
            h_a_vals[2] = raw_h_a.z * r_decay[2];
            h_a_vals[3] = raw_h_a.w * r_decay[3];
            h_b_vals[0] = raw_h_b.x * r_decay[0];
            h_b_vals[1] = raw_h_b.y * r_decay[1];
            h_b_vals[2] = raw_h_b.z * r_decay[2];
            h_b_vals[3] = raw_h_b.w * r_decay[3];
            dot_hk_a = h_a_vals[0] * r_k[0] + h_a_vals[1] * r_k[1] + h_a_vals[2] * r_k[2] + h_a_vals[3] * r_k[3];
            dot_hk_b = h_b_vals[0] * r_k[0] + h_b_vals[1] * r_k[1] + h_b_vals[2] * r_k[2] + h_b_vals[3] * r_k[3];

            const Sum2 dot_hk = warp_reduce_sum_pair(dot_hk_a, dot_hk_b);
            float const v_new0 = (s_v[v0] - dot_hk.x) * s_beta;
            float const v_new1 = (s_v[v1] - dot_hk.y) * s_beta;

            float dot_hq_a = 0.0f;
            float dot_hq_b = 0.0f;
            int64_t const state_head_offset
                = static_cast<int64_t>(slot) * state_slot_stride + static_cast<int64_t>(i_hv) * kDimV * kDimK;
            int64_t const state_idx_a = state_head_offset + static_cast<int64_t>(v0) * kDimK + k_base;
            int64_t const state_idx_b = state_head_offset + static_cast<int64_t>(v1) * kDimK + k_base;
            float const h_a_0 = h_a_vals[0] + r_k[0] * v_new0;
            float const h_a_1 = h_a_vals[1] + r_k[1] * v_new0;
            float const h_a_2 = h_a_vals[2] + r_k[2] * v_new0;
            float const h_a_3 = h_a_vals[3] + r_k[3] * v_new0;
            float const h_b_0 = h_b_vals[0] + r_k[0] * v_new1;
            float const h_b_1 = h_b_vals[1] + r_k[1] * v_new1;
            float const h_b_2 = h_b_vals[2] + r_k[2] * v_new1;
            float const h_b_3 = h_b_vals[3] + r_k[3] * v_new1;
            if constexpr (kComputeOutputBeforeStore)
            {
                dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
                dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
                store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
                store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
            }
            else
            {
                store_state_float4<kUseCacheGlobalStore>(state + state_idx_a, make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
                store_state_float4<kUseCacheGlobalStore>(state + state_idx_b, make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
                dot_hq_a = h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
                dot_hq_b = h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
            }

            const Sum2 dot_hq = warp_reduce_sum_pair(dot_hq_a, dot_hq_b);
            if (lane == 0)
            {
                s_o[v0] = dot_hq.x;
                s_o[v1] = dot_hq.y;
                if constexpr (kApplyOnorm && kAccumulateOnormSumsq)
                {
                    o_sumsq += dot_hq.x * dot_hq.x + dot_hq.y * dot_hq.y;
                }
            }
        }

        if constexpr (kPrefetchNextStateChunk)
        {
            if (chunk + 2 < kNumChunks)
            {
                cp_async_state_chunk(&s_state[0][0][0], state, slot, i_hv, state_slot_stride, chunk + 2);
            }
        }
    }
    __syncthreads();

    if constexpr (kApplyOnorm)
    {
        if constexpr (kAccumulateOnormSumsq)
        {
            if (lane == 0)
            {
                s_reduce[warp] = o_sumsq;
            }
            __syncthreads();

            float total_sumsq = 0.0f;
            if (warp == 0)
            {
                total_sumsq = lane < kWarps ? s_reduce[lane] : 0.0f;
                total_sumsq = warp_reduce_sum(total_sumsq);
                if (lane == 0)
                {
                    s_reduce[0] = total_sumsq;
                }
            }
            __syncthreads();

            if (tid < kDimV)
            {
                int64_t const out_idx = output_head * kDimV + tid;
                float const raw_o = s_o[tid];
                float const rstd = rsqrtf(s_reduce[0] / static_cast<float>(kDimV) + onorm_eps);
                float gate;
                float weight;
                if constexpr (kPreloadOnormParams)
                {
                    gate = pre_onorm_gate;
                    weight = pre_onorm_weight;
                }
                else
                {
                    gate = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + tid));
                    weight = onorm_weight[tid];
                }
                float const y = raw_o * rstd * weight * gate;
                out[out_idx] = bf16_store(y);
            }
        }
        else
        {
            float const raw_o = tid < kDimV ? s_o[tid] : 0.0f;
            float const o_sq = raw_o * raw_o;
            float sumsq;
            if constexpr (kUseActiveOnormReduction || kUseActiveQkReduction)
            {
                sumsq = block_reduce_sum_active_for<kDimV / 32>(o_sq, s_reduce);
            }
            else
            {
                sumsq = block_reduce_sum(o_sq, s_reduce);
            }

            if (tid < kDimV)
            {
                int64_t const out_idx = output_head * kDimV + tid;
                float const rstd = rsqrtf(sumsq / static_cast<float>(kDimV) + onorm_eps);
                float gate;
                float weight;
                if constexpr (kPreloadOnormParams)
                {
                    gate = pre_onorm_gate;
                    weight = pre_onorm_weight;
                }
                else
                {
                    gate = sigmoid_fast(bf16_load(onorm_g, i_n * layout.outputNormGateRowStride + i_hv * kDimV + tid));
                    weight = onorm_weight[tid];
                }
                float const y = raw_o * rstd * weight * gate;
                out[out_idx] = bf16_store(y);
            }
        }
    }
    else
    {
        if (tid < kDimV)
        {
            int64_t const out_idx = output_head * kDimV + tid;
            out[out_idx] = bf16_store(s_o[tid]);
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

struct KdaDecodeLaunchParams
{
    void const* x_q;
    void const* x_k;
    void const* x_v;
    void const* w_q_t;
    void const* w_k_t;
    void const* w_v_t;
    void const* bias_q;
    void const* bias_k;
    void const* bias_v;
    void* cs_q;
    void* cs_k;
    void* cs_v;
    float const* a_log;
    void const* g;
    float const* dt_bias;
    void const* beta;
    void const* onorm_g;
    float const* onorm_weight;
    int const* ssm_state_indices;
    int const* cu_seqlens;
    float* state;
    int64_t state_slot_stride;
    int64_t conv_state_slot_stride;
    void* out;
    int B;
    int H;
    int HV;
    bool apply_onorm;
    bool update_conv_cache;
    bool use_lower_bound;
    bool apply_beta_sigmoid;
    float lower_bound;
    float scale;
    float onorm_eps;
    KdaDecodeIoLayout layout;
    cudaStream_t stream;
};

template <int kHeads, bool kUseStaticDecodeLayout, bool kApplyOnorm, bool kUpdateConvState, bool kUseLowerBound,
    bool kApplyBetaSigmoid>
void launch_kda_decode_compact_heads_raw(KdaDecodeLaunchParams const& p)
{
    constexpr int kStageDynamicSmemBytes = 3 * 32 * kDimK * static_cast<int>(sizeof(float));
    auto const kernel = kda_decode_fusion_compact_heads_kernel<kApplyOnorm, true, kUseStaticDecodeLayout, kHeads,
        kHeads, false, false, false, true, false, true, kUpdateConvState, kUseLowerBound, kApplyBetaSigmoid>;
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kStageDynamicSmemBytes));
    common::launchWithPdlWhenEnabled("kdaDecodeCompactHeads", kernel, dim3(p.B * p.HV), dim3(kThreads),
        kStageDynamicSmemBytes, p.stream, reinterpret_cast<__nv_bfloat16 const*>(p.x_q),
        reinterpret_cast<__nv_bfloat16 const*>(p.x_k), reinterpret_cast<__nv_bfloat16 const*>(p.x_v),
        reinterpret_cast<__nv_bfloat16 const*>(p.w_q_t), reinterpret_cast<__nv_bfloat16 const*>(p.w_k_t),
        reinterpret_cast<__nv_bfloat16 const*>(p.w_v_t), reinterpret_cast<__nv_bfloat16 const*>(p.bias_q),
        reinterpret_cast<__nv_bfloat16 const*>(p.bias_k), reinterpret_cast<__nv_bfloat16 const*>(p.bias_v),
        reinterpret_cast<__nv_bfloat16*>(p.cs_q), reinterpret_cast<__nv_bfloat16*>(p.cs_k),
        reinterpret_cast<__nv_bfloat16*>(p.cs_v), p.a_log, reinterpret_cast<__nv_bfloat16 const*>(p.g), p.dt_bias,
        reinterpret_cast<__nv_bfloat16 const*>(p.beta), reinterpret_cast<__nv_bfloat16 const*>(p.onorm_g),
        p.onorm_weight, p.ssm_state_indices, p.cu_seqlens, p.state, p.state_slot_stride, p.conv_state_slot_stride,
        reinterpret_cast<__nv_bfloat16*>(p.out), p.B, p.H, p.HV, p.lower_bound, p.scale, p.onorm_eps, p.layout);
}

template <int kHeads, bool kUseStaticDecodeLayout, bool kApplyOnorm, bool kUpdateConvState, bool kUseLowerBound,
    bool kApplyBetaSigmoid>
void launch_kda_decode_many_heads_raw(KdaDecodeLaunchParams const& p)
{
    constexpr bool kUseHeadGrid = kUseStaticDecodeLayout;
    dim3 const grid = kUseStaticDecodeLayout ? dim3(p.B, p.HV) : dim3(p.B * p.HV);
    auto const kernel
        = kda_decode_fusion_many_heads_kernel<kApplyOnorm, kUseStaticDecodeLayout, kHeads, kHeads, kUseHeadGrid, false,
            false, false, false, false, true, true, true, kUpdateConvState, kUseLowerBound, kApplyBetaSigmoid>;
    common::launchWithPdlWhenEnabled("kdaDecodeManyHeads", kernel, grid, dim3(kThreads), 0, p.stream,
        reinterpret_cast<__nv_bfloat16 const*>(p.x_q), reinterpret_cast<__nv_bfloat16 const*>(p.x_k),
        reinterpret_cast<__nv_bfloat16 const*>(p.x_v), reinterpret_cast<__nv_bfloat16 const*>(p.w_q_t),
        reinterpret_cast<__nv_bfloat16 const*>(p.w_k_t), reinterpret_cast<__nv_bfloat16 const*>(p.w_v_t),
        reinterpret_cast<__nv_bfloat16 const*>(p.bias_q), reinterpret_cast<__nv_bfloat16 const*>(p.bias_k),
        reinterpret_cast<__nv_bfloat16 const*>(p.bias_v), reinterpret_cast<__nv_bfloat16*>(p.cs_q),
        reinterpret_cast<__nv_bfloat16*>(p.cs_k), reinterpret_cast<__nv_bfloat16*>(p.cs_v), p.a_log,
        reinterpret_cast<__nv_bfloat16 const*>(p.g), p.dt_bias, reinterpret_cast<__nv_bfloat16 const*>(p.beta),
        reinterpret_cast<__nv_bfloat16 const*>(p.onorm_g), p.onorm_weight, p.ssm_state_indices, p.cu_seqlens, p.state,
        p.state_slot_stride, p.conv_state_slot_stride, reinterpret_cast<__nv_bfloat16*>(p.out), p.B, p.H, p.HV,
        p.lower_bound, p.scale, p.onorm_eps, p.layout);
}

template <bool kCompact, int kHeads, bool kUseStaticDecodeLayout, bool kApplyOnorm, bool kUpdateConvState,
    bool kUseLowerBound, bool kApplyBetaSigmoid>
void launch_kda_decode_raw(KdaDecodeLaunchParams const& p)
{
    if constexpr (kCompact)
    {
        launch_kda_decode_compact_heads_raw<kHeads, kUseStaticDecodeLayout, kApplyOnorm, kUpdateConvState,
            kUseLowerBound, kApplyBetaSigmoid>(p);
    }
    else
    {
        launch_kda_decode_many_heads_raw<kHeads, kUseStaticDecodeLayout, kApplyOnorm, kUpdateConvState, kUseLowerBound,
            kApplyBetaSigmoid>(p);
    }
}

template <bool kCompact, int kHeads, bool kUseStaticDecodeLayout, bool kApplyOnorm, bool kUseLowerBound,
    bool kApplyBetaSigmoid>
void launch_kda_decode_selected_backend(KdaDecodeLaunchParams const& p)
{
    if (p.update_conv_cache)
    {
        launch_kda_decode_raw<kCompact, kHeads, kUseStaticDecodeLayout, kApplyOnorm, true, kUseLowerBound,
            kApplyBetaSigmoid>(p);
    }
    else
    {
        launch_kda_decode_raw<kCompact, kHeads, kUseStaticDecodeLayout, kApplyOnorm, false, kUseLowerBound,
            kApplyBetaSigmoid>(p);
    }
}

template <bool kCompact, int kHeads, bool kUseStaticDecodeLayout, bool kApplyOnorm, bool kUseLowerBound>
void dispatch_kda_decode_beta(KdaDecodeLaunchParams const& p)
{
    if (p.apply_beta_sigmoid)
    {
        launch_kda_decode_selected_backend<kCompact, kHeads, kUseStaticDecodeLayout, kApplyOnorm, kUseLowerBound, true>(
            p);
    }
    else
    {
        launch_kda_decode_selected_backend<kCompact, kHeads, kUseStaticDecodeLayout, kApplyOnorm, kUseLowerBound,
            false>(p);
    }
}

template <bool kCompact, int kHeads, bool kUseStaticDecodeLayout, bool kApplyOnorm>
void dispatch_kda_decode_decay(KdaDecodeLaunchParams const& p)
{
    if (p.use_lower_bound)
    {
        dispatch_kda_decode_beta<kCompact, kHeads, kUseStaticDecodeLayout, kApplyOnorm, true>(p);
    }
    else
    {
        dispatch_kda_decode_beta<kCompact, kHeads, kUseStaticDecodeLayout, kApplyOnorm, false>(p);
    }
}

template <bool kCompact, int kHeads, bool kUseStaticDecodeLayout>
void dispatch_kda_decode_features(KdaDecodeLaunchParams const& p)
{
    if (p.apply_onorm)
    {
        dispatch_kda_decode_decay<kCompact, kHeads, kUseStaticDecodeLayout, true>(p);
    }
    else
    {
        dispatch_kda_decode_decay<kCompact, kHeads, kUseStaticDecodeLayout, false>(p);
    }
}

template <bool kCompact, int kHeads>
void dispatch_kda_decode_layout(KdaDecodeLaunchParams const& p)
{
    if (p.ssm_state_indices == nullptr)
    {
        dispatch_kda_decode_features<kCompact, kHeads, true>(p);
    }
    else
    {
        dispatch_kda_decode_features<kCompact, kHeads, false>(p);
    }
}

template <bool kCompact>
void dispatch_kda_decode_heads(KdaDecodeLaunchParams const& p)
{
    switch (p.H)
    {
    case 1: dispatch_kda_decode_layout<kCompact, 1>(p); break;
    case 2: dispatch_kda_decode_layout<kCompact, 2>(p); break;
    case 3: dispatch_kda_decode_layout<kCompact, 3>(p); break;
    case 4: dispatch_kda_decode_layout<kCompact, 4>(p); break;
    case 6: dispatch_kda_decode_layout<kCompact, 6>(p); break;
    case 8: dispatch_kda_decode_layout<kCompact, 8>(p); break;
    case 12: dispatch_kda_decode_layout<kCompact, 12>(p); break;
    case 16: dispatch_kda_decode_layout<kCompact, 16>(p); break;
    case 24: dispatch_kda_decode_layout<kCompact, 24>(p); break;
    case 32: dispatch_kda_decode_layout<kCompact, 32>(p); break;
    case 48: dispatch_kda_decode_layout<kCompact, 48>(p); break;
    case 96: dispatch_kda_decode_layout<kCompact, 96>(p); break;
    default:
        if constexpr (kCompact)
        {
            TLLM_CHECK_WITH_INFO(false, "KDA compact-heads decode does not support numHeads=%d", p.H);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "KDA decode does not support numHeads=%d", p.H);
        }
    }
}

} // namespace

void invokeKdaDecode(KdaDecodeParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.numHeads == params.numValueHeads, "KDA decode requires numHeads == numValueHeads");
    bool const useCompactHeads = shouldUseCompactHeads(params.batchSize, params.numHeads, params.numValueHeads);
    KdaDecodeLaunchParams const launchParams{params.xQ, params.xK, params.xV, params.wQT, params.wKT, params.wVT,
        params.biasQ, params.biasK, params.biasV, params.convStateQ, params.convStateK, params.convStateV, params.logA,
        params.gate, params.dtBias, params.beta, params.outputNormGate, params.outputNormWeight, params.ssmStateIndices,
        params.cuSeqlens, params.state, params.stateSlotStride, params.convStateSlotStride, params.output,
        params.batchSize, params.numHeads, params.numValueHeads, params.applyOutputNorm, params.updateConvCache,
        params.useLowerBound, params.applyBetaSigmoid, params.lowerBound, params.scale, params.outputNormEps,
        params.layout, stream};
    if (useCompactHeads)
    {
        dispatch_kda_decode_heads<true>(launchParams);
    }
    else
    {
        dispatch_kda_decode_heads<false>(launchParams);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
