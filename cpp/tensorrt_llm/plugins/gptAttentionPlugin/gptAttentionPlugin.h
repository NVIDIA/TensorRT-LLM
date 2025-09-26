/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "checkMacrosPlugin.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{
// batch_size = num_ctx_requests + num_gen_requests * beam_width
// num_ctx_requests = number of context requests (single sequence per request).
// num_gen_requests = number of generation requests (beam_width sequences per request).
// Context sequences have to appear first, generation sequences after

// inputs (see GPTAttentionPlugin::isEntryUsed for when each tensor is actually used)
//     0.  input_tensor [batch_size, seq_len, local_hidden_size + 2 * local_num_kv_heads * head_size] or
//                      [num_tokens, local_hidden_size + 2 * local_num_kv_heads * head_size] when
//                      enable_remove_input_padding
//     1.  sequence_length [batch_size] (optional)
//     2.  host_past_key_value_lengths [batch_size] (int32) (optional)
//     3.  host_max_attention_window_sizes [num_layers] (int32)
//     4.  host_sink_token_length [1] (int32)
//     5.  context_lengths [batch_size]
//     6.  cache_indir [num_gen_requests, beam_width, memory_max_len] (required in beamsearch) (optional)
//     7.  host_request_types [batch_size] int32. 0: context; 1: generation: 2: none. When not in inflight-batching
//     mode,
//                      all elements must be identical.
//     8.  past_key_value_pool [batch_size, 2, local_num_kv_heads, max_seq_len, head_size] or
//         block_offsets [batch_size, 2, max_blocks_per_seq] if paged kv cache (optional)
//     8.1 host_block_offsets [batch_size, 2, max_blocks_per_seq] if paged kv cache (optional)
//     8.2 host_pool_pointers [2] if paged kv cache (optional)
//     9.  kv_cache_quantization_scale [1] (optional)
//     10. kv_cache_dequantization_scale [1] (optional)
//     11. attention_output_quantization_scale [1] (on device, optional)
//     12. attention_mask [num_tokens, kv_seqlen] (on device, bool, optional)
//     13. attention_packed_mask [num_tokens, kv_seqlen / 32] (on device, uint32_t, optional)
//          - pack masks by encoding multiple mask positions into a single 32-bit unsigned integer.
//          - see kernels/contextMultiHeadAttention/fmhaPackedMask.cpp for more details.
//     14. rotary_inv_freq [head_size / 2] or [head_size] (longrope type) (float) (on device, optional)
//     15. rotary_cos_sin [max_num_embedding_positions, 2] (float) (on device, optional)
//     16. alibi_slopes [num_heads] (optional for ALiBi position embedding)
//     17. relative_attention_bias [num_heads] (optional for ALiBi position embedding)
//     18. host_context_lengths [batch_size] int32. (optional, required when remove_input_padding is true)
//     19. qkv_bias (optional) [local_hidden_size * 3]
//     20. spec_decoding_generation_lengths (optional, required when medusa is enabled) (int32_t) [batch_size]
//     21. spec_decoding_packed_mask (optional, required when medusa is enabled) (int32_t) [num_tokens, packed_mask_dim]
//                                    packed_mask_dim = divUp(max_num_spec_decoding_tokens + 1, 32)
//     22. spec_decoding_position_offsets (optional, required when medusa is enabled) (int32_t) [batch_size,
//     max_num_spec_decoding_tokens + 1]
//     23. spec_decoding_use (optional, bool) [1]: If it is set as true, enable speculative decoding
//     24. long_rope_rotary_inv_freq [head / 2] (float) (on device, optional)
//     25. long_rope_rotary_cos_sin [max_num_embedding_positions, 2] (float) (on device, optional)
//     26. host_runtime_perf_knobs (int64)
//     27. host_context_progress (void*)
//     28. position_id_tensor(MLA) [total_tokens], used for rope embedding in MLA
//     29. q_a_proj_tensor(MLA) [hidden_dim, c_q_dim + c_k_dim + ropd_dim], used to proj compacted QKV
//     30. q_a_layernorm_tensor(MLA) [c_q_dim], rmsnorm weight for compacted q
//     31. q_b_proj_tensor(MLA) [c_q_dim, head_num * head_size], weight for companted q to q in context
//     32. kv_a_proj_with_mqa_tensor(MLA) [c_q_dim, head_num * (c_k_dim + rope_dim)], weight for companted q to kdim in
//     generation
//     33. kv_a_layernorm_tensor(MLA) [c_k_dim], rmsnorm weight for compacted kv
//     34. kv_b_proj_tensor(MLA) [c_k_dim, head_num * 2 * (head_size - rope_dim)], weight for compacted kv to kv in
//     context
//     35. skip_attn (optional, bool) [1]: If it is set as true, skip the atteniton plugin and return
//     directly.
//
// outputs
//     output_tensor [batch_size, seq_len, local_hidden_size]
//     present_key_value_pool (optional if not paged kv cache) [batch_size, 2, local_num_kv_heads, max_seq_len,
//     head_size]

class GPTAttentionPlugin : public GPTAttentionPluginCommon
{
public:
    GPTAttentionPlugin(int layer_idx, int num_heads, int vision_start, int vision_length, int num_kv_heads,
        int num_kv_heads_origin, int head_size, int unidirectional, float q_scaling, float attn_logit_softcapping_scale,
        tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim, // for RoPE. 0 for non-RoPE
        float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale,
        int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size,
        int tp_rank,           // for ALiBi
        bool unfuse_qkv_gemm,  // for AutoPP
        bool use_logn_scaling, // for LognScaling
        tensorrt_llm::kernels::ContextFMHAType context_fmha_type, int kv_cache_quant_mode, bool remove_input_padding,
        tensorrt_llm::kernels::AttentionMaskType mask_type,
        tensorrt_llm::kernels::BlockSparseParams block_sparse_params, bool paged_kv_cache, int tokens_per_block,
        nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled, bool cross_attention = false,
        int max_distance = 0, bool pos_shift_enabled = false, bool dense_context_fmha = false,
        bool use_paged_context_fmha = true, bool use_fp8_context_fmha = true, bool has_full_attention_mask = false,
        bool use_cache = true, bool is_spec_decoding_enabled = false,
        bool spec_decoding_is_generation_length_variable = false, int spec_decoding_max_generation_length = 1,
        bool is_mla_enabled = false, int q_lora_rank = 0, int kv_lora_rank = 0, int qk_nope_head_dim = 0,
        int qk_rope_head_dim = 0, int v_head_dim = 0, bool fuse_fp4_quant = false, bool skip_attn = false,
        int cp_size = 1, int cp_rank = 0, std::set<int32_t> cp_group = {});

    GPTAttentionPlugin(void const* data, size_t length);

    ~GPTAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    template <typename T, typename AttentionOutT, typename KVCacheBuffer>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    template <typename T, typename AttentionOutT = T>
    int enqueueDispatchKVCacheType(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    void configurePluginImpl(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept;
    template <typename T>
    void configurePluginDispatchKVCacheType(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    GPTAttentionPlugin* clone() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    template <typename T, typename AttentionOutT, typename KVCacheBuffer>
    int enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg, int32_t localNbTokens,
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    using IndexType = std::int32_t;

    std::vector<size_t> mEntryIdx;
    enum class IdxEntry : size_t
    {
        QKV_TENSOR,
        K_TENSOR,
        V_TENSOR,
        ATTENTION_MASK,
        ATTENTION_PACKED_MASK,
        SEQUENCE_LENGTH,
        HOST_PAST_KEY_VALUE_LENGTHS,
        HOST_MAX_ATTENTION_WINDOW,
        HOST_SINK_TOKEN_LENGTH,
        CONTEXT_LENGTHS,
        CACHE_INDIR,
        REQUEST_TYPES,
        KV_CACHE_BLOCK_OFFSETS,
        HOST_KV_CACHE_BLOCK_OFFSETS,
        HOST_KV_CACHE_POOL_POINTERS,
        HOST_KV_CACHE_POOL_MAPPING,
        PAST_KEY_VALUE,
        KV_CACHE_QUANTIZATION_SCALE,
        KV_CACHE_DEQUANTIZATION_SCALE,
        ATTENTION_OUTPUT_QUANTIZATION_SCALE,
        ATTENTION_OUTPUT_SF_SCALE,
        ROTARY_INV_FREQ,
        ROTARY_COS_SIN,
        ALIBI_SLOPES,
        RELATIVE_ATTENTION_BIAS,
        CROSS_KV,
        CROSS_KV_LENGTH,
        ENCODER_INPUT_LENGTH,
        HOST_CONTEXT_LENGTH,
        QKV_BIAS_TENSOR,
        SPEC_DECODING_GENERATION_LENGTHS,
        SPEC_DECODING_PACKED_MASK,
        SPEC_DECODING_POSITION_OFFSETS,
        SPEC_DECODING_USE,
        LONG_ROPE_ROTARY_INV_FREQ,
        LONG_ROPE_ROTARY_COS_SIN,
        MROPE_ROTARY_COS_SIN,
        MROPE_POSITION_DELTAS,
        HOST_RUNTIME_PERF_KNOBS,
        HOST_CONTEXT_PROGRESS,
        MLA_Q_B_PROJ_TENSOR,
        MLA_KV_B_PROJ_TENSOR,
        MLA_K_B_PROJ_TRANS_TENSOR,
        SKIP_ATTN,
        LOGN_SCALING,
        ENUM_SIZE, // Used to count the number of IdxEntry, must put in last
    };

    std::string toString(IdxEntry const& entry) const;
    bool isEntryUsed(IdxEntry const& entry) const;
    void initEntryIdx();
    IndexType getIdx(IdxEntry const& entry) const;

    // Get generation input sequence length (might be larger than 1 in the speculative decoding mode).
    int getGenerationInputSequenceLength(
        nvinfer1::PluginTensorDesc const* inputDesc, int32_t localNbSeq, int32_t localNbTokens) const;
};

class GPTAttentionPluginCreator : public GPTAttentionPluginCreatorCommon
{
public:
    GPTAttentionPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace tensorrt_llm::plugins
