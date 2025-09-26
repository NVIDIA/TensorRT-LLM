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
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <curand_kernel.h>

#include "static_switch.h"
#include <cuda.h>
#include <fmha/alibi_params.h>
#include <fused_multihead_attention_utils.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params
{
    // The QKV tensor.
    void* __restrict__ qkv_ptr;
    // The stride between rows of the Q, K and V matrices.
    size_t qkv_stride_in_bytes;
    // Number of attention heads.
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_fprop_params : public Qkv_params
{

    // The dQKV matrices.
    void* __restrict__ dqkv_ptr;

    // The dKV matrices.
    void* __restrict__ dkv_ptr;

    // The O matrix (output).
    void* __restrict__ o_ptr;

    // The dO matrix (output).
    void* __restrict__ do_ptr;

    // The matrix of softmax_lse = m + log(l), where m = rowmax(P), l = rowsum(exp(P - max))
    void* __restrict__ lse_ptr;

    // The matrix of softmax_sum, reduce_sum(dP * P) or reduce_sum (O * dP)
    void* __restrict__ softmax_sum_ptr;

    // The stride between rows of O.
    int64_t o_stride_in_bytes;

    // The pointer to the S matrix, overwritten by the dP matrix (bwd).
    void* __restrict__ s_ptr;
    // The stride between rows of the S matrix.
    int64_t s_stride_in_bytes;

    int64_t lse_stride_in_bytes;

    // The stride for softmax_sum
    int64_t sum_stride_in_bytes;

    // The dimensions.
    int b, s, d, total_s;

    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // scale_bmm1 in float
    float fscale_bmm1;

    // array of length b+1 holding starting offset of each sequence.
    int* __restrict__ cu_seqlens;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;

    // Scale factor of 2^16 * p_dropout in uint16_t
    uint16_t p_dropout_16bit;

    // Scale factor of 1 / (1 - p_dropout), in half2.
    uint32_t scale_dropout;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Randoms seeds ptr.
    void* __restrict__ seed_ptr;

    // flags for dP store
    bool save_dp;

    // flags for drop out
    bool has_dropout;

    // flag for bf16
    bool is_bf16;

    // flag for causal masking
    bool is_causal;

    // flag for Alibi Bias.
    bool has_alibi = false;
    fmha::AlibiParams alibi_params;

    // Temporary buffer (on the device) to compute dQ - uses FP32 atomics.
    void* dq_acc_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
struct Launch_params
{
    Launch_params(cudaDeviceProp* props_, cudaStream_t stream_, bool is_training_, bool is_causal_, bool is_nl_)
        : elts_per_thread(0)
        , props(props_)
        , stream(stream_)
        , is_training(is_training_)
        , is_causal(is_causal_)
        , is_nl(is_nl_)
    {
    }

    size_t elts_per_thread;

    cudaDeviceProp* props;

    cudaStream_t stream;

    bool is_training;

    bool is_causal;

    Kernel_params params;
    int num_full_heads;
    int num_main_groups;
    int heads_last_wave;
    int main_steps;
    int rest_steps;
    bool is_nl;
};
