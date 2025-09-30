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
namespace fprop
{

struct Fmha_fprop_params
{

    Fmha_fprop_params(void* qkv_ptr_, void* o_ptr_, float* m_ptr_, float* zi_ptr_, int* cu_seqlens_, float* amax_s_,
        float* amax_o_, int b_, int s_, int h_, int d_, float p_dropout, float scale_bmm_q_k_, float* ptr_d_scale_qkv_,
        float* ptr_d_scale_s_, float* ptr_d_scale_o_, float* ptr_q_scale_s_, float* ptr_q_scale_o_,
        uint64_t* ptr_philox_unpacked_)
        : qkv_ptr(qkv_ptr_)
        , o_ptr(o_ptr_)
        , m_ptr(m_ptr_)
        , zi_ptr(zi_ptr_)
        , cu_seqlens(cu_seqlens_)
        , amax_s(amax_s_)
        , amax_o(amax_o_)
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
        , ptr_q_scale_s(ptr_q_scale_s_)
        , ptr_q_scale_o(ptr_q_scale_o_)
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
        this->s_stride_in_bytes = b * h * s * sizeof(typename Traits::Accumulator_type);
    }

    // I N P U T S.
    // Packed query key value tensors.
    void* __restrict__ qkv_ptr;
    // Softmax "statistics": row-wise max.
    float* __restrict__ m_ptr;
    // Softmax "statistics": row-wise normalizer.
    float* __restrict__ zi_ptr;
    // Self-attention mask information.
    int* __restrict__ cu_seqlens;

    // O U T P U T S.
    // Output of the forward pass.
    void* __restrict__ o_ptr;
    // Amax for dqkv output.
    float* __restrict__ amax_o;
    // Amax for dqkv output.
    float* __restrict__ amax_s;
    // The memory containing the data for Philox generation.
    uint64_t* __restrict__ ptr_philox_unpacked;

    // D E B U G.
    void* __restrict__ s_ptr = nullptr;
    void* __restrict__ p_ptr = nullptr;
    void* __restrict__ print_buf = nullptr;

    // These depend on kernel traits and will be configured by launcher.
    size_t qkv_stride_in_bytes = 0;
    size_t o_stride_in_bytes = 0;

    size_t s_stride_in_bytes = 0;

    int b;
    int s;
    int h;
    int d;

    // D R O P O U T.
    float p_keep;
    float rp_keep;
    at::PhiloxCudaState philox_args;

    // A T T E N T I O N.
    float scale_bmm_q_k; // attention scale  P = scale * Q x K'

    // U N S C A L E   F A C T O R S.
    // Input unscale factor.
    float* ptr_d_scale_qkv;
    // Output unscale factors.
    float* ptr_d_scale_s;
    float* ptr_d_scale_o;

    // S C A L E   F A C T O R S.
    // Input scale factors.
    float* ptr_q_scale_s;
    float* ptr_q_scale_o;
};

template <typename Params_>
struct Launch_params_
{
    using Params = Params_;

    Launch_params_(cudaDeviceProp* props_, cudaStream_t stream_, Params params_, bool is_training_)
        : props(props_)
        , stream(stream_)
        , params(params_)
        , is_training(is_training_)
    {
    }

    cudaDeviceProp* props;
    cudaStream_t stream;
    Params params;
    // Upper bound for standard launch: ceildiv(seq_len^2, THREADS_PER_CTA)
    size_t elts_per_thread;
    bool is_training;

    void init_philox_state(at::CUDAGeneratorImpl* gen)
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(elts_per_thread);
    }
};

using Launch_params = Launch_params_<Fmha_fprop_params>;

void run_fmha_fprop_fp8_512_64_sm90(Launch_params& launch_params, bool const configure);

} // namespace fprop
} // namespace hopper
} // namespace fmha
