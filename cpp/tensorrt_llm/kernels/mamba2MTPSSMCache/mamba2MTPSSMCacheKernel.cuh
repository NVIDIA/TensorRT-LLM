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

#pragma once

#include "mamba2MTPSSMCache.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// each warp 32 threads hold the entire ssm_dim
constexpr int MTP_NUM_WARPS = 4;
constexpr int MTP_WARP_SIZE = 32;
constexpr int MTP_NUM_BLOCK_THREADS = MTP_NUM_WARPS * MTP_WARP_SIZE;
constexpr int MTP_HDIMS_PER_WARP = 2; // do not change

#define MTP_DISPATCH_BOOL(VALUE, NAME, ...)                                                                            \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (VALUE)                                                                                                     \
        {                                                                                                              \
            constexpr bool NAME = true;                                                                                \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            constexpr bool NAME = false;                                                                               \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define MTP_DISPATCH_DTYPE(DTYPE, NAME, ...)                                                                           \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (DTYPE == Mamba2Dtype::kBFloat16)                                                                           \
        {                                                                                                              \
            using NAME = nv_bfloat16;                                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (DTYPE == Mamba2Dtype::kFloat16)                                                                       \
        {                                                                                                              \
            using NAME = __half;                                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (DTYPE == Mamba2Dtype::kFloat32)                                                                       \
        {                                                                                                              \
            using NAME = float;                                                                                        \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

template <typename T>
__device__ __forceinline__ float mtp_to_float(T val);

template <>
__device__ __forceinline__ float mtp_to_float<float>(float val)
{
    return val;
}

template <>
__device__ __forceinline__ float mtp_to_float<nv_bfloat16>(nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <>
__device__ __forceinline__ float mtp_to_float<__half>(__half val)
{
    return __half2float(val);
}

template <typename T>
__device__ __forceinline__ T mtp_from_float(float val);

template <>
__device__ __forceinline__ float mtp_from_float<float>(float val)
{
    return val;
}

template <>
__device__ __forceinline__ nv_bfloat16 mtp_from_float<nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ __half mtp_from_float<__half>(float val)
{
    return __float2half(val);
}

__device__ __forceinline__ float mtp_thresholded_softplus(float dt_value)
{
    constexpr float threshold = 20.f;
    return (dt_value <= threshold) ? __logf(1.f + __expf(dt_value)) : dt_value;
}

__device__ __forceinline__ float mtp_warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int VEC_SIZE, typename T>
__device__ __forceinline__ void mtp_load_vec_to_float(float* dst, T const* src);

template <int VEC_SIZE>
__device__ __forceinline__ void mtp_load_vec_to_float(float* dst, nv_bfloat16 const* src)
{
    if constexpr (VEC_SIZE == 4)
    {
        uint2 raw;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n" : "=r"(raw.x), "=r"(raw.y) : "l"(src));
        nv_bfloat162 const* bf16_pairs = reinterpret_cast<nv_bfloat162 const*>(&raw);
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float2 f2 = __bfloat1622float2(bf16_pairs[i]);
            dst[2 * i + 0] = f2.x;
            dst[2 * i + 1] = f2.y;
        }
    }
    else
    {
#pragma unroll
        for (int c = 0; c < VEC_SIZE / 8; ++c)
        {
            uint4 raw;
            asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                         : "l"(src + c * 8));
            nv_bfloat162 const* bf16_pairs = reinterpret_cast<nv_bfloat162 const*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 f2 = __bfloat1622float2(bf16_pairs[i]);
                dst[c * 8 + 2 * i + 0] = f2.x;
                dst[c * 8 + 2 * i + 1] = f2.y;
            }
        }
    }
}

template <int VEC_SIZE>
__device__ __forceinline__ void mtp_load_vec_to_float(float* dst, __half const* src)
{
    if constexpr (VEC_SIZE == 4)
    {
        uint2 raw;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n" : "=r"(raw.x), "=r"(raw.y) : "l"(src));
        __half2 const* fp16_pairs = reinterpret_cast<__half2 const*>(&raw);
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float2 f2 = __half22float2(fp16_pairs[i]);
            dst[2 * i + 0] = f2.x;
            dst[2 * i + 1] = f2.y;
        }
    }
    else
    {
#pragma unroll
        for (int c = 0; c < VEC_SIZE / 8; ++c)
        {
            uint4 raw;
            asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                         : "l"(src + c * 8));
            __half2 const* fp16_pairs = reinterpret_cast<__half2 const*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 f2 = __half22float2(fp16_pairs[i]);
                dst[c * 8 + 2 * i + 0] = f2.x;
                dst[c * 8 + 2 * i + 1] = f2.y;
            }
        }
    }
}

template <int VEC_SIZE>
__device__ __forceinline__ void mtp_load_vec_to_float(float* dst, float const* src)
{
#pragma unroll
    for (int c = 0; c < VEC_SIZE / 4; ++c)
    {
        float4 vec;
        asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                     : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w)
                     : "l"(src + c * 4));
        dst[c * 4 + 0] = vec.x;
        dst[c * 4 + 1] = vec.y;
        dst[c * 4 + 2] = vec.z;
        dst[c * 4 + 3] = vec.w;
    }
}

template <int VEC_SIZE, typename T>
__device__ __forceinline__ void mtp_float_save_vec_to(T* dst, float const* src);

template <int VEC_SIZE>
__device__ __forceinline__ void mtp_float_save_vec_to(nv_bfloat16* dst, float const* src)
{
    if constexpr (VEC_SIZE == 4)
    {
        uint2 raw;
        nv_bfloat162* bf16_pairs = reinterpret_cast<nv_bfloat162*>(&raw);
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float2 f2 = make_float2(src[2 * i + 0], src[2 * i + 1]);
            bf16_pairs[i] = __float22bfloat162_rn(f2);
        }
        __stcs(reinterpret_cast<uint2*>(dst), raw);
    }
    else
    {
#pragma unroll
        for (int c = 0; c < VEC_SIZE / 8; ++c)
        {
            uint4 raw;
            nv_bfloat162* bf16_pairs = reinterpret_cast<nv_bfloat162*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 f2 = make_float2(src[c * 8 + 2 * i + 0], src[c * 8 + 2 * i + 1]);
                bf16_pairs[i] = __float22bfloat162_rn(f2);
            }
            __stcs(reinterpret_cast<uint4*>(dst + c * 8), raw);
        }
    }
}

template <int VEC_SIZE>
__device__ __forceinline__ void mtp_float_save_vec_to(__half* dst, float const* src)
{
    if constexpr (VEC_SIZE == 4)
    {
        uint2 raw;
        __half2* fp16_pairs = reinterpret_cast<__half2*>(&raw);
#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            float2 f2 = make_float2(src[2 * i + 0], src[2 * i + 1]);
            fp16_pairs[i] = __float22half2_rn(f2);
        }
        __stcs(reinterpret_cast<uint2*>(dst), raw);
    }
    else
    {
#pragma unroll
        for (int c = 0; c < VEC_SIZE / 8; ++c)
        {
            uint4 raw;
            __half2* fp16_pairs = reinterpret_cast<__half2*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 f2 = make_float2(src[c * 8 + 2 * i + 0], src[c * 8 + 2 * i + 1]);
                fp16_pairs[i] = __float22half2_rn(f2);
            }
            __stcs(reinterpret_cast<uint4*>(dst + c * 8), raw);
        }
    }
}

template <int VEC_SIZE>
__device__ __forceinline__ void mtp_float_save_vec_to(float* dst, float const* src)
{
#pragma unroll
    for (int c = 0; c < VEC_SIZE / 4; ++c)
    {
        float4 vec;
        vec.x = src[c * 4 + 0];
        vec.y = src[c * 4 + 1];
        vec.z = src[c * 4 + 2];
        vec.w = src[c * 4 + 3];
        __stcs(reinterpret_cast<float4*>(dst + c * 4), vec);
    }
}

template <int VEC_SIZE, bool HAS_D, bool HAS_Z, bool HAS_DT_BIAS, bool DT_SOFTPLUS, bool RETRIEVE_PARENT_TOKEN,
    typename ssm_D, typename in_out_D, typename weight_D, typename A_D>
__global__ void __launch_bounds__(MTP_NUM_BLOCK_THREADS)
    mamba2MTPSSMCacheKernel(ssm_D* state, in_out_D const* x, weight_D const* dt, A_D const* A, in_out_D const* B,
        in_out_D const* C, in_out_D* out, ssm_D* intermediate_states, weight_D const* D, in_out_D const* z,
        weight_D const* dt_bias, int32_t const* ssm_batch_indices, int32_t const* intermediate_ssm_batch_indices,
        int32_t const* retrieve_parent_token, int const cache_steps, bool const has_ssm_batch_indices,
        bool const has_intermediate_ssm_batch_indices, int const pad_slot_id, bool const disable_state_update,
        int const nheads, int const head_dim, int const ssm_dim, int const heads_groups_ratio,
        int const stride_nheads_hdim_ssm_dim, int const stride_hdim_ssm_dim, int const stride_cache_nheads_hdim,
        int const stride_nheads_hdim, int const stride_cache_ngroups_ssm_dim, int const stride_ngroups_ssm_dim)
{
    int const tid = threadIdx.x;
    int const lane_id = tid & 31;
    int const warp_id = tid >> 5;
    int const head_id = blockIdx.y;
    int const bs_id = blockIdx.z;
    int const hdim_id_a = blockIdx.x * MTP_NUM_WARPS * MTP_HDIMS_PER_WARP + warp_id * MTP_HDIMS_PER_WARP;
    int const hdim_id_b = hdim_id_a + 1;
    int const nheads_hdim_id_a = head_id * head_dim + hdim_id_a;
    int const nheads_hdim_id_b = nheads_hdim_id_a + 1;

    int state_idx = has_ssm_batch_indices ? ssm_batch_indices[bs_id] : bs_id;
    if (state_idx == pad_slot_id)
    {
        return;
    }

    int intermediate_state_idx = has_intermediate_ssm_batch_indices ? intermediate_ssm_batch_indices[bs_id] : state_idx;

    float state_4_a[VEC_SIZE], state_4_b[VEC_SIZE];
    mtp_load_vec_to_float<VEC_SIZE>(state_4_a,
        state + state_idx * stride_nheads_hdim_ssm_dim + head_id * stride_hdim_ssm_dim + hdim_id_a * ssm_dim
            + lane_id * VEC_SIZE);
    mtp_load_vec_to_float<VEC_SIZE>(state_4_b,
        state + state_idx * stride_nheads_hdim_ssm_dim + head_id * stride_hdim_ssm_dim + hdim_id_b * ssm_dim
            + lane_id * VEC_SIZE);

    float A_val = 1.0f;
    float dt_bias_val = 0.0f;
    float D_val = 1.0f;
    if (!lane_id)
    {
        A_val = mtp_to_float<A_D>(A[head_id]);
        if constexpr (HAS_DT_BIAS)
        {
            dt_bias_val = mtp_to_float<weight_D>(dt_bias[head_id]);
        }
        if constexpr (HAS_D)
        {
            D_val = mtp_to_float<weight_D>(D[head_id]);
        }
    }

    in_out_D const* B_base
        = B + bs_id * stride_cache_ngroups_ssm_dim + head_id / heads_groups_ratio * ssm_dim + lane_id * VEC_SIZE;
    in_out_D const* C_base
        = C + bs_id * stride_cache_ngroups_ssm_dim + head_id / heads_groups_ratio * ssm_dim + lane_id * VEC_SIZE;

    ssm_D* inter_base_a = intermediate_states + intermediate_state_idx * stride_nheads_hdim_ssm_dim * cache_steps
        + head_id * stride_hdim_ssm_dim + hdim_id_a * ssm_dim + lane_id * VEC_SIZE;
    ssm_D* inter_base_b = intermediate_states + intermediate_state_idx * stride_nheads_hdim_ssm_dim * cache_steps
        + head_id * stride_hdim_ssm_dim + hdim_id_b * ssm_dim + lane_id * VEC_SIZE;

#pragma unroll
    for (int t = 0; t < cache_steps; ++t)
    {
        if constexpr (RETRIEVE_PARENT_TOKEN)
        {
            if (t != 0)
            {
                int parent_step_idx = retrieve_parent_token[bs_id * cache_steps + t];
                if (parent_step_idx >= 0 && parent_step_idx < cache_steps)
                {
                    mtp_load_vec_to_float<VEC_SIZE>(
                        state_4_a, inter_base_a + parent_step_idx * stride_nheads_hdim_ssm_dim);
                    mtp_load_vec_to_float<VEC_SIZE>(
                        state_4_b, inter_base_b + parent_step_idx * stride_nheads_hdim_ssm_dim);
                }
            }
        }

        float B_4[VEC_SIZE];
        float C_4[VEC_SIZE];
        mtp_load_vec_to_float<VEC_SIZE>(B_4, B_base + t * stride_ngroups_ssm_dim);
        mtp_load_vec_to_float<VEC_SIZE>(C_4, C_base + t * stride_ngroups_ssm_dim);

        float x_val_a, x_val_b, xdt_val_a, xdt_val_b, dA_val;
        if (!lane_id)
        {
            float dt_val = mtp_to_float<weight_D>(dt[bs_id * cache_steps * nheads + t * nheads + head_id]);
            int const base = bs_id * stride_cache_nheads_hdim + t * stride_nheads_hdim;
            x_val_a = mtp_to_float<in_out_D>(x[base + nheads_hdim_id_a]);
            x_val_b = mtp_to_float<in_out_D>(x[base + nheads_hdim_id_b]);
            if constexpr (HAS_DT_BIAS)
            {
                dt_val += dt_bias_val;
            }
            if constexpr (DT_SOFTPLUS)
            {
                dt_val = mtp_thresholded_softplus(dt_val);
            }
            dA_val = __expf(A_val * dt_val);
            xdt_val_a = x_val_a * dt_val;
            xdt_val_b = x_val_b * dt_val;
        }
        dA_val = __shfl_sync(0xffffffff, dA_val, 0);

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            state_4_a[i] *= dA_val;
            state_4_b[i] *= dA_val;
        }

        xdt_val_a = __shfl_sync(0xffffffff, xdt_val_a, 0);
        xdt_val_b = __shfl_sync(0xffffffff, xdt_val_b, 0);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            state_4_a[i] = __fmaf_rn(B_4[i], xdt_val_a, state_4_a[i]);
            state_4_b[i] = __fmaf_rn(B_4[i], xdt_val_b, state_4_b[i]);
        }

        mtp_float_save_vec_to<VEC_SIZE>(inter_base_a + t * stride_nheads_hdim_ssm_dim, state_4_a);
        mtp_float_save_vec_to<VEC_SIZE>(inter_base_b + t * stride_nheads_hdim_ssm_dim, state_4_b);

        float out_val_a = 0.0f;
        float out_val_b = 0.0f;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            out_val_a = __fmaf_rn(state_4_a[i], C_4[i], out_val_a);
            out_val_b = __fmaf_rn(state_4_b[i], C_4[i], out_val_b);
        }

        out_val_a = mtp_warp_reduce_sum(out_val_a);
        out_val_b = mtp_warp_reduce_sum(out_val_b);

        if (!lane_id)
        {
            int const out_base = bs_id * stride_cache_nheads_hdim + t * stride_nheads_hdim;
            if constexpr (HAS_D)
            {
                out_val_a = __fmaf_rn(x_val_a, D_val, out_val_a);
                out_val_b = __fmaf_rn(x_val_b, D_val, out_val_b);
            }
            if constexpr (HAS_Z)
            {
                float z_val_a = mtp_to_float<in_out_D>(z[out_base + nheads_hdim_id_a]);
                float z_val_b = mtp_to_float<in_out_D>(z[out_base + nheads_hdim_id_b]);
                out_val_a *= z_val_a * __frcp_rn(1.0f + __expf(-z_val_a));
                out_val_b *= z_val_b * __frcp_rn(1.0f + __expf(-z_val_b));
            }

            out[out_base + nheads_hdim_id_a] = mtp_from_float<in_out_D>(out_val_a);
            out[out_base + nheads_hdim_id_b] = mtp_from_float<in_out_D>(out_val_b);
        }
    }

    if (!disable_state_update)
    {
        mtp_float_save_vec_to<VEC_SIZE>(state + state_idx * stride_nheads_hdim_ssm_dim + head_id * stride_hdim_ssm_dim
                + hdim_id_a * ssm_dim + lane_id * VEC_SIZE,
            state_4_a);
        mtp_float_save_vec_to<VEC_SIZE>(state + state_idx * stride_nheads_hdim_ssm_dim + head_id * stride_hdim_ssm_dim
                + hdim_id_b * ssm_dim + lane_id * VEC_SIZE,
            state_4_b);
    }
}

// Host-side launch function template — dispatches dtype + bool combinations and launches kernel.
// Each VEC_SIZE is explicitly instantiated in its own .cu file for parallel compilation.
template <int VEC_SIZE>
void launchMamba2MTPSSMCacheKernel(Mamba2MTPSSMCacheParams const& params, dim3 grid, dim3 block, cudaStream_t stream)
{
    bool has_D = (params.D != nullptr);
    bool has_z = (params.z != nullptr);
    bool has_dt_bias = (params.dt_bias != nullptr);
    bool has_ssm_indices = (params.ssm_batch_indices != nullptr);
    bool has_inter_indices = (params.intermediate_states_indices != nullptr);
    bool has_parent_token = (params.retrieve_parent_token != nullptr);

    int const nheads = params.nheads;
    int const head_dim = params.head_dim;
    int const ssm_dim = params.ssm_dim;
    int const ngroups = params.ngroups;
    int const heads_groups_ratio = nheads / ngroups;

    MTP_DISPATCH_DTYPE(params.ssm_dtype, ssm_D,
        [&]
        {
            MTP_DISPATCH_DTYPE(params.in_out_dtype, in_out_D,
                [&]
                {
                    MTP_DISPATCH_DTYPE(params.weight_dtype, weight_D,
                        [&]
                        {
                            MTP_DISPATCH_DTYPE(params.a_dtype, A_D,
                                [&]
                                {
                                    MTP_DISPATCH_BOOL(has_D, BOOL_HAS_D,
                                        [&]
                                        {
                                            MTP_DISPATCH_BOOL(has_z, BOOL_HAS_Z,
                                                [&]
                                                {
                                                    MTP_DISPATCH_BOOL(has_dt_bias, BOOL_HAS_DT_BIAS,
                                                        [&]
                                                        {
                                                            MTP_DISPATCH_BOOL(params.dt_softplus, BOOL_DT_SOFTPLUS,
                                                                [&]
                                                                {
                                                                    MTP_DISPATCH_BOOL(has_parent_token,
                                                                        BOOL_PARENT_TOKEN,
                                                                        [&]
                                                                        {
                                                                            mamba2MTPSSMCacheKernel<VEC_SIZE,
                                                                                BOOL_HAS_D, BOOL_HAS_Z,
                                                                                BOOL_HAS_DT_BIAS, BOOL_DT_SOFTPLUS,
                                                                                BOOL_PARENT_TOKEN, ssm_D, in_out_D,
                                                                                weight_D,
                                                                                A_D><<<grid, block, 0, stream>>>(
                                                                                reinterpret_cast<ssm_D*>(params.ssm),
                                                                                reinterpret_cast<in_out_D const*>(
                                                                                    params.x),
                                                                                reinterpret_cast<weight_D const*>(
                                                                                    params.dt),
                                                                                reinterpret_cast<A_D const*>(params.A),
                                                                                reinterpret_cast<in_out_D const*>(
                                                                                    params.B),
                                                                                reinterpret_cast<in_out_D const*>(
                                                                                    params.C),
                                                                                reinterpret_cast<in_out_D*>(params.out),
                                                                                reinterpret_cast<ssm_D*>(
                                                                                    params.intermediate_states),
                                                                                BOOL_HAS_D
                                                                                    ? reinterpret_cast<weight_D const*>(
                                                                                        params.D)
                                                                                    : nullptr,
                                                                                BOOL_HAS_Z
                                                                                    ? reinterpret_cast<in_out_D const*>(
                                                                                        params.z)
                                                                                    : nullptr,
                                                                                BOOL_HAS_DT_BIAS
                                                                                    ? reinterpret_cast<weight_D const*>(
                                                                                        params.dt_bias)
                                                                                    : nullptr,
                                                                                params.ssm_batch_indices,
                                                                                params.intermediate_states_indices,
                                                                                BOOL_PARENT_TOKEN
                                                                                    ? params.retrieve_parent_token
                                                                                    : nullptr,
                                                                                params.cache_steps, has_ssm_indices,
                                                                                has_inter_indices, params.pad_slot_id,
                                                                                params.disable_state_update, nheads,
                                                                                head_dim, ssm_dim, heads_groups_ratio,
                                                                                nheads * head_dim * ssm_dim,
                                                                                head_dim * ssm_dim,
                                                                                params.cache_steps * nheads * head_dim,
                                                                                nheads * head_dim,
                                                                                params.cache_steps * ngroups * ssm_dim,
                                                                                ngroups * ssm_dim);
                                                                        });
                                                                });
                                                        });
                                                });
                                        });
                                });
                        });
                });
        });
}

} // namespace kernels

TRTLLM_NAMESPACE_END
