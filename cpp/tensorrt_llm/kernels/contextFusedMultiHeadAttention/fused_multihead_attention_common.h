/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
// The number of positions in one uint32_t.
static constexpr int NUM_POSITIONS_IN_UINT32 = 32;
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
    // Causal mask + attend to the specific sliding window or chunk.
    SLIDING_OR_CHUNKED_CAUSAL,
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
    Q_PAGED_KV,
    // Q has contiguous [B, S, H, D] layout, while K has contiguous [B, S, H_kv, D] layout, and V has
    // contiguous [B, S, H_kv, D_v] layout. Only used for context MLA now.
    SEPARATE_Q_K_V,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MHARunnerFixedParams
{
    // The FMHA input data type.
    Data_type dataType;
    // The FMHA kv cache data type.
    Data_type dataTypeKv;
    // The FMHA data output type.
    Data_type dataTypeOut;

    // Do we use fp32 accumulation ?
    // TODO: remove forceFp32Acc from MHARunnerFixedParams after adding host_runtime_perf_knobs to
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
    // The number of tokens per kv cache block.
    int numTokensPerBlock;
    // The head size.
    int headSize;
    // The head size of V.
    int headSizeV = 0;
    // The head size of Q/K non-RoPE part, only used for MLA now.
    int headSizeQkNope = 0;
    // The scaling applied to bmm1_scale.
    float qScaling;
    // The attention logit softcapping scale.
    float attnLogitSoftcappingScale;
    // Do we apply alibi ?
    bool hasAlibi;
    // Scale the alibi bias or not ?
    bool scaleAlibi;
    // save softmax stats?
    bool saveSoftmax;
    // The tensor parallel size (alibi).
    int tpSize = 1;
    // The tensor parallel rank (alibi).
    int tpRank = 0;
    // q tensor quant block size in sage attention
    int sageBlockSizeQ = 0;
    // k tensor quant block size in sage attention
    int sageBlockSizeK = 0;
    // v tensor quant block size in sage attention
    int sageBlockSizeV = 0;

    // Convert to string for debug.
    std::string convertToStrOutput()
    {
        std::string output = "dataType = ";
        output += data_type_to_string(dataType);

        output += ", dataTypeKv = ";
        output += data_type_to_string(dataTypeKv);

        output += ", dataTypeOut = ";
        output += data_type_to_string(dataTypeOut);

        output += ", forceFp32Acc = " + std::string(forceFp32Acc ? "true" : "false");

        output += ", attentionMaskType = ";
        switch (attentionMaskType)
        {
        case ContextAttentionMaskType::PADDING: output += "padding"; break;
        case ContextAttentionMaskType::CAUSAL: output += "causal"; break;
        case ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL: output += "sliding_or_chunked_causal"; break;
        case ContextAttentionMaskType::CUSTOM_MASK: output += "custom_mask"; break;
        default: output += std::to_string(static_cast<int>(attentionMaskType)) + " (unknown)"; break;
        }

        output += ", attentionInputLayout = ";
        switch (attentionInputLayout)
        {
        case AttentionInputLayout::PACKED_QKV: output += "packed_qkv"; break;
        case AttentionInputLayout::Q_CONTIGUOUS_KV: output += "q_contiguous_kv"; break;
        case AttentionInputLayout::Q_PAGED_KV: output += "q_paged_kv"; break;
        case AttentionInputLayout::SEPARATE_Q_K_V: output += "separate_q_k_v"; break;
        default: output += std::to_string(static_cast<int>(attentionInputLayout)) + " (unknown)"; break;
        }

        output += ", isSPadded = " + std::string(isSPadded ? "true" : "false");
        output += ", numQHeads = " + std::to_string(numQHeads);
        output += ", numKvHeads = " + std::to_string(numKvHeads);
        output += ", numTokensPerBlock = " + std::to_string(numTokensPerBlock);
        output += ", headSize = " + std::to_string(headSize);
        output += ", headSizeV = " + std::to_string(headSizeV);
        output += ", qScaling = " + std::to_string(qScaling);
        output += ", attnLogitSoftcappingScale = " + std::to_string(attnLogitSoftcappingScale);
        output += ", hasAlibi = " + std::string(hasAlibi ? "true" : "false");
        output += ", scaleAlibi = " + std::string(scaleAlibi ? "true" : "false");
        output += ", tpSize = " + std::to_string(tpSize);
        output += ", tpRank = " + std::to_string(tpRank);
        output += ", sageBlockSizeQ = " + std::to_string(sageBlockSizeQ);
        output += ", sageBlockSizeK = " + std::to_string(sageBlockSizeK);
        output += ", sageBlockSizeV = " + std::to_string(sageBlockSizeV);

        return output;
    }

    /**
     * Set attention mask type from AttentionMaskType enum
     * @param maskType The AttentionMaskType to use
     * @return Reference to this object for method chaining
     * @throws If the maskType cannot be mapped to ContextAttentionMaskType
     */
    MHARunnerFixedParams& setAttentionMaskType(std::int8_t maskType)
    {
        switch (maskType)
        {
        case 0: // tensorrt_llm::kernels::AttentionMaskType::PADDING
            attentionMaskType = ContextAttentionMaskType::PADDING;
            break;
        case 1: // tensorrt_llm::kernels::AttentionMaskType::CAUSAL
            attentionMaskType = ContextAttentionMaskType::CAUSAL;
            break;
        case 2: // tensorrt_llm::kernels::AttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL
            attentionMaskType = ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL;
            break;
        // NOTE: For BIDIRECTIONAL, BIDIRECTIONALGLM, BLOCKSPARSE context phase, CAUSAL mask is used
        case 3: // tensorrt_llm::kernels::AttentionMaskType::BIDIRECTIONAL
            attentionMaskType = ContextAttentionMaskType::CAUSAL;
            break;
        case 4: // tensorrt_llm::kernels::AttentionMaskType::BIDIRECTIONALGLM
            attentionMaskType = ContextAttentionMaskType::CAUSAL;
            break;
        case 5: // tensorrt_llm::kernels::AttentionMaskType::BLOCKSPARSE
            attentionMaskType = ContextAttentionMaskType::CAUSAL;
            break;
        case 6: // tensorrt_llm::kernels::AttentionMaskType::CUSTOM_MASK
            attentionMaskType = ContextAttentionMaskType::CUSTOM_MASK;
            break;
        default:
            TLLM_THROW("AttentionMaskType %d cannot be mapped to ContextAttentionMaskType", static_cast<int>(maskType));
        }
        return *this;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MHARunnerParams
{
    // The batch size.
    int b;
    // The number of grouped heads.
    int numGroupedHeads = 1;
    // The max q sequence length.
    int qSeqLen;
    // The max kv sequence length.
    int kvSeqLen;
    // The sliding window size.
    int slidingWindowSize;
    // The chunked attention size.
    int chunkedAttentionSize = INT_MAX;
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
    // The K buffer ptr (for separate K input).
    void const* kPtr;
    // The V buffer ptr (for separate V input).
    void const* vPtr;
    // The paged kv cache array.
    KVBlockArray pagedKvCache;
    // The paged kv cache array for scaling factor.
    KVBlockArray pagedKvSfCache;
    // The output buffer ptr.
    void* outputPtr;
    // The output scaling factor buffer ptr. (only used for FP4 output)
    void* outputSfPtr;
    // The softmax_status ptr for RingAttention.
    void* softmaxStatsPtr;
    // The attention sinks ptr.
    float const* attentionSinksPtr;
    // The packed mask ptr.
    void const* packedMaskPtr;
    // The cumulative Q sequence lengths.
    void const* cuQSeqLenPtr;
    // The KV sequence lengths.
    void const* kvSeqLenPtr;
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
    // The device scale for O scaling factor.
    float const* oSfScalePtr;
    // The cuda stream.
    cudaStream_t stream;
    // Force using fp32 accumulation data type.
    bool forceFp32Acc = false;
    // pointer to q, k, v scale tensor in sageattention
    float* qScalePtr;
    float* kScalePtr;
    float* vScalePtr;
    // q, k, v block size in sageattention
    int qMaxNBlock;
    int kMaxNBlock;
    int vMaxNBlock;
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
    // The separate K matrice.
    void const* k_ptr;
    // The separate V matrice.
    void const* v_ptr;
    // The separate KV matrice.
    void const* kv_ptr;
    // The separate paged kv cache.
    KVBlockArrayForContextFMHA paged_kv_cache;
    // The mask to implement drop-out.
    void const* packed_mask_ptr;
    // The attention sinks.
    float const* attention_sinks_ptr;
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
    cudaTmaDesc tma_desc_q;
    // Tma descriptors for packed/contiguous/paged kv cache.
    // Kv in packed qkv layout: [B, S, 3, H, D]
    // Contiguous kv layout: [B, 2, H, S, D].
    // Paged kv layout: [UINT32_MAX, H, Tokens_per_block, D].
    cudaTmaDesc tma_desc_k;
    cudaTmaDesc tma_desc_v;
    // Tma descriptor for o
    cudaTmaDesc tma_desc_o;

    // Tma load of paged kv cache.
    int blocks_per_tma_load;
    int blocks_per_tma_load_log2;

    // The dimensions. In ordinary multi-head attention (MHA), there are equal number of QKV heads
    int b, h, h_kv, h_q_per_kv, s, d;
    // The dimension of V. If unset, dv = d.
    int dv = 0;
    // The number of grouped heads.
    int num_grouped_heads = 1;
    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    int sliding_window_size = INT_MAX;
    // The chunked attention size in log2 (> 0 means chunked attention is used)
    int log2_chunked_attention_size = 0;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, softcapping_scale_bmm1, scale_softmax, scale_bmm2;

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

    // SageAttention parameters
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
    // padded head size for tma descriptors.
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
    // enable attention logit softcapping scale.
    bool enableAttnLogitSoftcapping = false;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
    // total device memory (used by TMA loading of paged kv cache).
    size_t total_device_memory = 0;
    // q tensor quant block size in sage attention
    int sage_block_size_q = 0;
    // k tensor quant block size in sage attention
    int sage_block_size_k = 0;
    // v tensor quant block size in sage attention
    int sage_block_size_v = 0;
    // if we use a kernel that supports returning softmax statistics
    bool supportReturnSoftmaxStats;
};

} // namespace kernels
} // namespace tensorrt_llm
