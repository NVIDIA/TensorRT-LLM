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

#include <cuda.h>
#include <fmha/alibi_params.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/paged_kv_cache.h>
#include <fused_multihead_attention_utils.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Make sure the mask input is padded to 128 x 256 tile size in order to
// match all Ampere/Hopper kernels.
static constexpr int FLASH_ATTEN_MASK_M_ALIGNMENT = 128;
static constexpr int FLASH_ATTEN_MASK_N_ALIGNMENT = 256;
// The packed mask's MMA tile size is 64 x 64.
static constexpr int FLASH_ATTEN_MASK_MMA_M = 64;
static constexpr int FLASH_ATTEN_MASK_MMA_N = 64;

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Attention_mask_type
{
    // Mask the padded tokens.
    PADDING = 0,
    // Mask the padded tokens and all the tokens that come after in a sequence.
    CAUSAL,
    // Causal mask + attend to the specific sliding window or chunk.
    SLIDING_OR_CHUNKED_CAUSAL,
    // The custom mask input.
    CUSTOM_MASK,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline std::string mask_type_to_string(Attention_mask_type mask_type)
{
    switch (mask_type)
    {
    case Attention_mask_type::PADDING: return "padding";
    case Attention_mask_type::CAUSAL: return "causal";
    case Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL: return "sliding_or_chunked_causal";
    case Attention_mask_type::CUSTOM_MASK: return "custom_mask";
    default: assert(false); return "";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Attention_input_layout
{
    // QKV are packed into [B, S, 3, H, D] layout.
    PACKED_QKV = 0,
    // Q has contiguous [B, S, H, D] layout, while KV has contiguous [B, 2, H, S, D] layout.
    CONTIGUOUS_Q_KV,
    // Q has contiguous [B, S, H, D] layout, while paged KV layout are blocks of indices with shape
    // of [B, 2, Blocks_per_Seq], and the indice indicates the block distance to the pool ptr in
    // global memory.
    Q_PAGED_KV,
    // Q has [B, S, H, D] layout,
    // K has [B, S, H_kv, D] layout,
    // V has [B, S, H_kv, Dv] layout,
    SEPARATE_Q_K_V,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline std::string attention_input_layout_to_string(Attention_input_layout layout)
{
    switch (layout)
    {
    case Attention_input_layout::PACKED_QKV: return "packed_qkv";
    case Attention_input_layout::CONTIGUOUS_Q_KV: return "contiguous_q_kv";
    case Attention_input_layout::Q_PAGED_KV: return "contiguous_q_paged_kv";
    case Attention_input_layout::SEPARATE_Q_K_V: return "separate_q_k_v";
    default: assert(false); return "";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace bert
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#if USE_DEMO_BERT_PARAMS

// TODO TRT plugins use a different parameter struct taken from the old XMMA fork.
//      Until all cubins in the plugin are replaced with new kernels, we need to conform to that.
#include <fused_multihead_attention_demo_bert_params.h>

#else
struct Fused_multihead_attention_params_base
{
    // The QKV matrices.
    void* qkv_ptr;
    // The O matrix (output).
    void* o_ptr;

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

#if defined(DEBUG_HAS_PRINT_BUFFER)
    void* print_ptr;
#endif

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;
    // The bmm2 scaling factors in the device.
    uint32_t* scale_bmm1_d;
    uint32_t* scale_bmm2_d;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    bool enable_i2f_trick;

    // true: for int8, instead of doing max reduce, use max value encoded in scale factor
    bool use_int8_scale_max = false;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    fmha::AlibiParams alibi_params;

    // The number of heads computed by one iteration of the wave.
    int heads_per_wave;
    // Buffers to perform a global sync and a critical section.
    int *counters, *max_barriers, *sum_barriers, *locks;
    // Scratch buffers to finalize softmax.
    float *max_scratch_ptr, *sum_scratch_ptr;
    // Scratch buffer to finalize the output (not needed for FP16).
    int* o_scratch_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v1 : Fused_multihead_attention_params_base
{
    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The mask to implement drop-out.
    void* packed_mask_ptr;

    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v2 : Fused_multihead_attention_params_base
{
    // The dimension of V. If unset, dv = d.
    int dv = 0;

    // The input to support any mask patterns.
    void* packed_mask_ptr;
    // The mask input's stride in the N (K-seq) dimension.
    int64_t packed_mask_stride_in_bytes;
    // The Softmax stats vector of layout [total_tokens_q, h, 2], including softmax_max and softmax_sum
    void* softmax_stats_ptr;
    // The stride between rows of softmax_stats_ptr, default: h * sizeof(float2)
    int64_t softmax_stats_stride_in_bytes;

    // The attention sinks (per head).
    float* attention_sinks;

    // array of length b+1 holding prefix sum of actual q sequence lengths.
    int* cu_q_seqlens;
    // array of length b+1 holding prefix sum of actual kv sequence lengths.
    int* cu_kv_seqlens;
    // array of length b+1 holding prefix sum of actual mask sequence lengths.
    // it might not be the same as cu_q_seqlens as the mask seqlens will be padded.
    int* cu_mask_rows;

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

    // Contiguous Q buffer pointer [B, S, H, D].
    void* q_ptr;
    // The separate K matrice.
    void* k_ptr;
    // The separate V matrice.
    void* v_ptr;
    // Contiguous KV buffer pointer [B, 2, H, S, D].
    void* kv_ptr;
    // Paged KV Cache buffer.
    fmha::Kv_block_array paged_kv_cache;
    // Q and KV stride (used by LDGSTS).
    int64_t q_stride_in_bytes;
    int64_t k_stride_in_bytes;
    int64_t v_stride_in_bytes;

    // Paged KV load.
    int blocks_per_tma_load;
    int blocks_per_tma_load_log2;

    // M tile id counter for dynamic scheduling
    uint32_t* tile_id_counter_ptr;
    uint32_t num_tiles;
    uint32_t num_tiles_per_head;
    bool use_balanced_scheduling;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    int h_kv = 0;
    // h_q_per_kv is sometimes rematerialized in the kernel by formula h / h_kv to reclaim one register
    int h_q_per_kv = 1;

    // The number of grouped heads in the seqlen dimension.
    int num_grouped_heads = 1;

    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;

    // The chunked attention size (<= 0 means no chunked attention).
    int log2_chunked_attention_size = 0;

    // The softcapping scale (scale * tanh (x / scale)) applied to bmm1 output.
    float softcapping_scale_bmm1 = 0.0f;

    // is input/output padded
    bool is_s_padded = false;

    struct SageAttention
    {
        struct Scales
        {
            // this field is only used in bin/fmha.exe, will be omitted in exported cubin
            int block_size;
            // ceil(max_seqlen / block_size)
            int max_nblock;
            // The scale of each block, layout: (B, H, max_nblock)
            float* scales;
        } q, k, v;
    } sage;
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// flags to control kernel choice
struct Fused_multihead_attention_launch_params
{
    // flags to control small batch kernel choice
    // true: never unroll
    bool ignore_b1opt = false;
    // true: always unroll
    bool force_unroll = false;
    // use fp32 accumulation
    bool force_fp32_acc = false;
    // the C/32 format
    bool interleaved = false;
    // by default TMA is not used.
    bool use_tma = false;
    // total number of q tokens to set tma descriptors
    int total_q_seqlen = 0;
    // total number of kv tokens to set tma descriptors
    int total_kv_seqlen = 0;
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool use_granular_tiling = false;
    // causal masking or sliding_or_chunked_causal masking or dense(padding) mask.
    fmha::Attention_mask_type attention_mask_type = fmha::Attention_mask_type::PADDING;
    // the attention input layout.
    fmha::Attention_input_layout attention_input_layout = fmha::Attention_input_layout::PACKED_QKV;
    // enable_attn_logit_softcapping (choose kernels with softcapping_scale_bmm1).
    bool enable_attn_logit_softcapping = false;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace bert
