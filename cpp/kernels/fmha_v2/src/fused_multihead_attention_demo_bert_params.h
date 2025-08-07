/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmha/alibi_params.h>
#include <fmha/hopper/tma_types.h>
#include <limits.h>

struct Fused_multihead_attention_params_v1
{
    // The QKV matrices.
    void* qkv_ptr;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif // defined(STORE_S)

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    bool enable_i2f_trick;

    // true: for int8, instead of doing max reduce, use max value encoded in scale factor
    bool use_int8_scale_max = false;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    fmha::AlibiParams alibi_params{};

    // The number of heads computed by one iteration of the wave.
    int heads_per_wave;
    // Buffers to perform a global sync and a critical section.
    int *counters, *max_barriers, *sum_barriers, *locks;
    // Scratch buffers to finalize softmax.
    float *max_scratch_ptr, *sum_scratch_ptr;
    // Scratch buffer to finalize the output (not needed for FP16).
    int* o_scratch_ptr;
};

struct Fused_multihead_attention_params_v2
{
    // The packed QKV matrices.
    void* qkv_ptr;
    // The separate Q matrice.
    void* q_ptr;
    // The separate K matrice.
    void* k_ptr;
    // The separate V matrice.
    void* v_ptr;
    // The separate KV matrice (contiguous KV).
    void* kv_ptr;
    // The separate paged kv cache.
    fmha::Kv_block_array paged_kv_cache;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The attention sinks (per head).
    float* attention_sinks;
    // The O matrix (output).
    void* o_ptr;
    // The Softmax stats vector of layout [2, B, S, H], including softmax_sum and softmax_max
    void* softmax_stats_ptr;

    // The stride between rows of Q.
    int64_t q_stride_in_bytes;
    // The stride between rows of K.
    int64_t k_stride_in_bytes;
    // The stride between rows of V.
    int64_t v_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;
    // The stride between rows of softmax_stats_ptr
    int64_t softmax_stats_stride_in_bytes;

    // tma descriptors on device.
    // Either q in packed qkv [B, S, 3, H, D] of separate q layout [B, S, H, D].
    fmha::cudaTmaDesc tma_desc_q;
    // Tma descriptors for packed/contiguous/paged kv cache.
    // Kv in packed qkv layout: [B, S, 3, H, D]
    // Contiguous kv layout: [B, 2, H, S, D].
    // Paged kv layout: [UINT32_MAX, H, Tokens_per_block, D].
    fmha::cudaTmaDesc tma_desc_k;
    fmha::cudaTmaDesc tma_desc_v;
    // Tma descriptor for o
    fmha::cudaTmaDesc tma_desc_o;

    // Tma load of paged kv cache.
    int blocks_per_tma_load;
    int blocks_per_tma_load_log2;

    // The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads
    int b, h, h_kv, h_q_per_kv, s, d;
    // The dimension of V. If unset, dv = d.
    int dv = 0;
    // The number of grouped heads in the seqlen dimension.
    int num_grouped_heads = 1;
    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;
    // The chunked attention size in log2 (> 0 means that chunked attention is enabled).
    int log2_chunked_attention_size = 0;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, softcapping_scale_bmm1, scale_softmax, scale_bmm2;

    // The scaling factors in the device memory (required by TRT-LLM + FP8 FMHA).
    uint32_t* scale_bmm1_d;
    uint32_t* scale_bmm2_d;

    // array of length b+1 holding prefix sum of actual q sequence lengths.
    int* cu_q_seqlens;
    // array of length b+1 holding prefix sum of actual kv sequence lengths.
    int* cu_kv_seqlens;
    // array of length b+1 holding prefix sum of actual mask sequence lengths.
    // it might not be the same as cu_q_seqlens as the mask seqlens will be padded.
    int* cu_mask_rows;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    fmha::AlibiParams alibi_params{};

    // M tile id counter for dynamic scheduling
    uint32_t* tile_id_counter_ptr;
    uint32_t num_tiles;
    uint32_t num_tiles_per_head;
    bool use_balanced_scheduling;

    // is input/output padded
    bool is_s_padded = false;

    struct SageAttention
    {
        struct Scales
        {
            // ceil(max_seqlen / block_size)
            int max_nblock;
            // The scale of each block, layout: (B, H, max_nblock)
            float* scales;
        } q, k, v;
    } sage;
};
