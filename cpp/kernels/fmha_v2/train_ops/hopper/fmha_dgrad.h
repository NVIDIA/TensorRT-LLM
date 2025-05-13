/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace fmha
{
namespace hopper
{
namespace dgrad
{

struct Fmha_dgrad_params
{

    Fmha_dgrad_params(void* dqkv_ptr_, void* qkv_ptr_, void* do_ptr_, void* o_ptr_, float* m_ptr_, float* zi_ptr_,
        int* cu_seqlens_, float* amax_dp_, float* amax_dqkv_, void* dq_tmp_ptr_, int b_, int s_, int h_, int d_,
        float p_dropout, float scale_bmm_q_k_, float* ptr_d_scale_qkv_, float* ptr_d_scale_s_, float* ptr_d_scale_o_,
        float* ptr_d_scale_do_, float* ptr_d_scale_dp_, float* ptr_d_scale_dqkv_, float* ptr_q_scale_s_,
        float* ptr_q_scale_dp_, float* ptr_q_scale_dqkv_, uint64_t* ptr_philox_unpacked_)
        : dqkv_ptr(dqkv_ptr_)
        , qkv_ptr(qkv_ptr_)
        , do_ptr(do_ptr_)
        , o_ptr(o_ptr_)
        , m_ptr(m_ptr_)
        , zi_ptr(zi_ptr_)
        , cu_seqlens(cu_seqlens_)
        , amax_dp(amax_dp_)
        , amax_dqkv(amax_dqkv_)
        , dq_tmp_ptr(dq_tmp_ptr_)
        , b(b_)
        , s(s_)
        , h(h_)
        , d(d_)
        , p_keep(1.f - p_dropout)
        , rp_keep(1.f / (1.f - p_dropout))
        , scale_bmm_q_k(scale_bmm_q_k_)
        , ptr_d_scale_qkv(ptr_d_scale_qkv_)
        , ptr_d_scale_s(ptr_d_scale_s_)
        , ptr_d_scale_o(ptr_d_scale_o_)
        , ptr_d_scale_do(ptr_d_scale_do_)
        , ptr_d_scale_dp(ptr_d_scale_dp_)
        , ptr_d_scale_dqkv(ptr_d_scale_dqkv_)
        , ptr_q_scale_s(ptr_q_scale_s_)
        , ptr_q_scale_dp(ptr_q_scale_dp_)
        , ptr_q_scale_dqkv(ptr_q_scale_dqkv_)
        , ptr_philox_unpacked(ptr_philox_unpacked_)
    {
    }

    template <typename Kernel_traits>
    void set_strides()
    {
        using Traits = typename Kernel_traits::Traits_p;
        this->qkv_stride_in_bytes = h * 3 * d * sizeof(typename Traits::A_type);
        this->o_stride_in_bytes = h * d * sizeof(typename Traits::C_type);
        static_assert(sizeof(typename Traits::A_type) == sizeof(typename Traits::C_type));
        this->ds_stride_in_bytes = b * h * s * sizeof(typename Traits::Accumulator_type);
    }

    // I N P U T S.
    // Packed query key value tensors.
    void* __restrict__ qkv_ptr;
    // Softmax "statistics": row-wise max.
    float* __restrict__ m_ptr;
    // Softmax "statistics": row-wise normalizer.
    float* __restrict__ zi_ptr;
    // Output of the forward pass.
    void* __restrict__ o_ptr;
    // Incoming gradient dL/dO.
    void* __restrict__ do_ptr;
    // Self-attention mask information.
    int* __restrict__ cu_seqlens;

    // O U T P U T S.
    // dQKV.
    void* __restrict__ dqkv_ptr;
    // Intermediate dq for accumulation.
    void* __restrict__ dq_tmp_ptr;
    // Amax for dqkv output.
    float* __restrict__ amax_dqkv;
    // Amax for dqkv output.
    float* __restrict__ amax_dp;

    // D E B U G.
    void* __restrict__ s_ptr = nullptr;
    void* __restrict__ ds_ptr = nullptr;
    void* __restrict__ dp_ptr = nullptr;

    void* __restrict__ print_buf = nullptr;

    // These depend on kernel traits and will be configured by launcher.
    size_t qkv_stride_in_bytes = 0;
    size_t o_stride_in_bytes = 0;

    size_t ds_stride_in_bytes = 0;

    int b;
    int s;
    int h;
    int d;

    // D R O P O U T.
    float p_keep;
    float rp_keep;
    at::PhiloxCudaState philox_args;

    float scale_bmm_q_k; // attention scale  P = Q  x K'

    // U N S C A L E   F A C T O R S.
    // Input unscale factors.
    float* ptr_d_scale_qkv;
    float* ptr_d_scale_s;
    float* ptr_d_scale_o;
    float* ptr_d_scale_do;

    // Output unscale factors.
    float* ptr_d_scale_dp;
    float* ptr_d_scale_dqkv;

    // S C A L E   F A C T O R S.
    // Input scale factors.
    float* ptr_q_scale_s;
    float* ptr_q_scale_dp;
    float* ptr_q_scale_dqkv;

    // The Philox data.
    uint64_t* ptr_philox_unpacked;
};

template <typename Params_>
struct Launch_params_
{
    using Params = Params_;

    Launch_params_(cudaDeviceProp* props_, cudaStream_t stream_, Params params_, bool all_e5m2_ = false)
        : props(props_)
        , stream(stream_)
        , params(params_)
        , all_e5m2(all_e5m2_)
    {
    }

    cudaDeviceProp* props;
    cudaStream_t stream;
    Params params;
    // Upper bound for standard launch: ceildiv(seq_len^2, THREADS_PER_CTA)
    size_t elts_per_thread;

    void init_philox_state(at::CUDAGeneratorImpl* gen)
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(elts_per_thread);
    }

    bool all_e5m2;
};

using Launch_params = Launch_params_<Fmha_dgrad_params>;

void run_fmha_dgrad_fp8_512_64_sm90(Launch_params& launch_params, bool const configure);

} // namespace dgrad
} // namespace hopper
} // namespace fmha
