/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tmaDescriptor.h"
#include <limits.h>
#include <stdint.h>

#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// compute groups for warp-specialized kernels on Hopper
static constexpr int NUM_COMPUTE_GROUPS = 2;

// Make sure the packed mask input is padded to 128 x 256 tile size in order to
// match all Ampere/Hopper kernels.
static constexpr int FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT = 128;
static constexpr int FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT = 256;
// The packed mask's MMA tile size is 64 x 64.
static constexpr int FLASH_ATTEN_PACKED_MASK_MMA_M = 64;
static constexpr int FLASH_ATTEN_PACKED_MASK_MMA_N = 64;
// The flash attention always uses 4x1 warp layout.
static constexpr int FLASH_ATTEN_WARPS_M = 4;
static constexpr int FLASH_ATTEN_WARPS_N = 1;
// The number of threads per warp group.
static constexpr int NUM_THREADS_PER_WARP_GROUP = FLASH_ATTEN_WARPS_M * FLASH_ATTEN_WARPS_N * 32;
// The number of core mmas_n in one uint32_t packed mask.
static constexpr int NUM_CORE_MMAS_N = FLASH_ATTEN_PACKED_MASK_MMA_N / 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class ContextFMHAType
{
    DISABLED,
    ENABLED,
    // FP32 accumulation (FP16 I/O)
    ENABLED_WITH_FP32_ACC
};

enum class ContextAttentionMaskType
{
    // Mask the padded tokens.
    PADDING = 0,
    // Mask the padded tokens and all the tokens that come after in a sequence.
    CAUSAL,
    // Causal mask + mask the beginning tokens based on sliding_window_size
    // Only pay attention to [max(0, q_i - sliding_window_size), q_i]
    SLIDING_WINDOW_CAUSAL,
    // The custom mask input.
    CUSTOM_MASK
};

enum class AttentionInputLayout
{
    // QKV are packed into [B, S, 3, H, D] layout.
    PACKED_QKV = 0,
    // Q has contiguous [B, S, H, D] layout, while KV has contiguous [B, 2, H, S, D] layout.
    Q_CONTIGUOUS_KV,
    // Q has contiguous [B, S, H, D] layout, while paged KV has [B, 2, Max_blocks_per_seq] layout
    // that contains paged block indices. The indices indicate the block offset to the pool ptr in
    // global memory
    Q_PAGED_KV
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MHARunnerFixedParams
{
    // The FMHA data type.
    Data_type dataType;
    // Do we use fp32 accumulation ?
    // TODO(yibinl): remove forceFp32Acc from MHARunnerFixedParams after adding host_runtime_perf_knobs to
    // bertAttentionPlugin input tensors, so that we can change mLaunchParams.force_fp32_acc value in runtime.
    bool forceFp32Acc;
    // The attention mask type.
    ContextAttentionMaskType attentionMaskType;
    // The attention input layout.
    AttentionInputLayout attentionInputLayout;
    // Are the sequences in the batch padded ?
    bool isSPadded;
    // The number of Q heads.
    int numQHeads;
    // The number of Kv Heads.
    int numKvHeads;
    // The head size.
    int headSize;
    // The scaling applied to bmm1_scale.
    float qScaling;
    // The tanh scale after bmm1 (used in Grok models).
    float qkTanhScale;
    // Do we apply alibi ?
    bool hasAlibi;
    // Scale the alibi bias or not ?
    bool scaleAlibi;
    // The tensor parallel size (alibi).
    int tpSize = 1;
    // The tensor parallel rank (alibi).
    int tpRank = 0;

    // Convert to string for debug.
    std::string convertToStrOutput()
    {
        // Data type.
        std::string output = "data_type = ";
        switch (dataType)
        {
        case DATA_TYPE_FP16: output += forceFp32Acc ? "fp16_fp32" : "fp16"; break;
        case DATA_TYPE_BF16: output += "bf16"; break;
        case DATA_TYPE_E4M3: output += "e4m3"; break;
        default: TLLM_CHECK_WITH_INFO(false, "not supported.");
        }
        // Head size.
        output += ", head_size = " + std::to_string(headSize);
        // Attention mask type.
        output += ", attention_mask_type = ";
        switch (attentionMaskType)
        {
        case ContextAttentionMaskType::PADDING: output += "padding"; break;
        case ContextAttentionMaskType::CAUSAL: output += "causal"; break;
        case ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL: output += "sliding_window_causal"; break;
        case ContextAttentionMaskType::CUSTOM_MASK: output += "custom_mask"; break;
        default: TLLM_CHECK_WITH_INFO(false, "not supported.");
        }
        // Attention mask type.
        output += ", attention_input_layout = ";
        switch (attentionInputLayout)
        {
        case AttentionInputLayout::PACKED_QKV: output += "packed_qkv"; break;
        case AttentionInputLayout::Q_CONTIGUOUS_KV: output += "q_contiguous_kv"; break;
        case AttentionInputLayout::Q_PAGED_KV: output += "q_paged_kv"; break;
        default: TLLM_CHECK_WITH_INFO(false, "not supported.");
        }
        // Alibi.
        output += ", alibi = ";
        output += (hasAlibi ? "true" : "false");
        // QK tanh scale.
        output += ", qk_tanh_scale = ";
        output += (qkTanhScale != 0.f ? "true" : "false");

        return output;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MHARunnerParams
{
    // The batch size.
    int b;
    // The max q sequence length.
    int qSeqLen;
    // The max kv sequence length.
    int kvSeqLen;
    // The sliding window size.
    int slidingWindowSize;
    // The total number of Q sequence lengths in the batch.
    int totalQSeqLen;
    // The total number of KV sequence lengths in the batch.
    int totalKvSeqLen;

    // Buffers.
    // The packed QKV buffer ptr.
    void const* qkvPtr;
    // The Q buffer ptr.
    void const* qPtr;
    // The contiguous Kv buffer ptr;
    void const* kvPtr;
    // The paged kv cache array.
    KVBlockArray pagedKvCache;
    // The output buffer ptr.
    void* outputPtr;
    // The packed mask ptr.
    void const* packedMaskPtr;
    // The cumulative Q sequence lengths.
    void const* cuQSeqLenPtr;
    // The cumulative KV sequence lengths.
    void const* cuKvSeqLenPtr;
    // The cumulative packed mask rows.
    void const* cuMaskRowsPtr;
    // The dynamic scheduler tile counter.
    void* tileCounterPtr;
    // The bmm1 scale device ptr (only used by fp8 kernels).
    float const* scaleBmm1Ptr;
    // The bmm2 scale device ptr (only used by fp8 kernels).
    float const* scaleBmm2Ptr;
    // The cuda stream.
    cudaStream_t stream;
    bool forceFp32Acc = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct AlibiParams
{
    constexpr static int round_down_to_power_two(int x)
    {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    AlibiParams() = default;

    AlibiParams(int h, float scale_after_alibi)
        : scale_after_alibi(scale_after_alibi)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }

    AlibiParams(int h, int s, int tp_size, int rank, float scale_after_alibi)
        : AlibiParams(h * tp_size, scale_after_alibi)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};
    float alibi_neg4_div_h{};
    float scale_after_alibi{};
    // Could be simplified to `int rank` derive the others as `num_heads * rank, s * rank` at
    // runtime, but this makes assumptions about the layout downstream
    // (e.g. downstream may only split across the head dimension, so s would be the full sequence)
    int head_idx_offset = 0;
    int sequence_pos_offset = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v2
{
    // The QKV matrices.
    void const* qkv_ptr;
    // The separate Q matrice.
    void const* q_ptr;
    // The separate KV matrice.
    void const* kv_ptr;
    // The separate paged kv cache.
    KVBlockArrayForContextFMHA paged_kv_cache;
    // The mask to implement drop-out.
    void const* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between rows of the separate Q matrice.
    int64_t q_stride_in_bytes;
    // The stride between rows of the separate KV matrice.
    int64_t kv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

    // tma descriptors on device.
    // Either q in packed qkv [B, S, 3, H, D] of separate q layout [B, S, H, D].
    cudaTmaDesc tma_desc_q;
    // Tma descriptors for packed/contiguous/paged kv cache.
    // Kv in packed qkv layout: [B, S, 3, H, D]
    // Contiguous kv layout: [B, 2, H, S, D].
    // Paged kv layout: [UINT32_MAX, H, Tokens_per_block, D].
    cudaTmaDesc tma_desc_kv;
    // Tma descriptor for o
    cudaTmaDesc tma_desc_o;

    // Tma load of paged kv cache.
    int blocks_per_tma_load;
    int blocks_per_tma_load_log2;

    // The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads
    int b, h, h_kv, h_q_per_kv, s, d;
    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, tanh_scale_bmm1, scale_softmax, scale_bmm2;

    // The scaling factors in the device memory.
    uint32_t const* scale_bmm1_d;
    uint32_t const* scale_bmm2_d;

    // array of length b+1 holding prefix sum of actual q sequence lengths.
    int const* cu_q_seqlens;
    // array of length b+1 holding prefix sum of actual kv sequence lengths.
    int const* cu_kv_seqlens;
    // array of length b+1 holding prefix sum of actual mask sequence lengths.
    // it might not be the same as cu_q_seqlens as the mask seqlens will be padded.
    int const* cu_mask_rows;

    // If the kernel is using alibi or not
    bool has_alibi = false;
    AlibiParams alibi_params{};

    // M tile id counter for dynamic scheduling
    uint32_t* tile_id_counter_ptr;
    uint32_t num_tiles;
    uint32_t num_tiles_per_head;
    bool use_balanced_scheduling;

    // is input/output padded
    bool is_s_padded = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// flags to control kernel choice
struct Launch_params
{
    // seq_length to select the kernel
    int kernel_s = 0;
    // total q sequence length (considering the paddings).
    int total_q_seqlen = 0;
    // total kv sequence length.
    int total_kv_seqlen = 0;
    // padded head size (new power of 2) for tma descriptors.
    int padded_d = 0;
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
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool granular_tiling = false;
    // dynamic tile scheduling.
    bool dynamic_scheduler = false;
    // mask type: padding, causal, sliding_window_causal, custom_mask.
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    // input layout: packed_qkv, q_contiguous_kv, q_paged_kv.
    AttentionInputLayout attention_input_layout = AttentionInputLayout::PACKED_QKV;
    // use specialized kernels without alibi support.
    bool useKernelWithoutAlibi = false;
    // enable exp2 optimization (which helps improve performance).
    // note that this is not compatible with alibi bias due to the accuracy issues.
    bool useBase2ExpTrick = false;
    // enable scale + tanh for qk products.
    bool enableQKTanhScale = false;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
    // total device memory (used by TMA loading of paged kv cache).
    size_t total_device_memory = 0;
};

} // namespace kernels
} // namespace tensorrt_llm
