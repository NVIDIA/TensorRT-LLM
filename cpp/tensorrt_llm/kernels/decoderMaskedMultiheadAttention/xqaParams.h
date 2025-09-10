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
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

using XQADataType = Data_type;

struct XQAParams
{
    XQADataType data_type = DATA_TYPE_FP16;
    XQADataType kv_cache_data_type = DATA_TYPE_FP16;
    XQADataType output_data_type = DATA_TYPE_FP16;
    void* output = nullptr;
    void* output_sf = nullptr;
    void const* qkv = nullptr;
    int32_t const* cache_indir = nullptr;
    float const* attention_sinks = nullptr;
    float const* kv_scale_orig_quant = nullptr;
    float const* kv_scale_quant_orig = nullptr;
    int32_t const* host_past_key_value_lengths = nullptr;
    int32_t const* host_context_lengths = nullptr;
    int32_t* semaphores = nullptr;
    void* workspaces = nullptr;
    uint32_t batch_size = 0;
    int32_t beam_width = 0;
    int32_t chunked_attention_size = INT_MAX;
    int32_t max_attention_window_size = 0;
    int32_t cyclic_attention_window_size = 0;
    int32_t sink_token_length = 0;
    int max_past_kv_length = 0;
    void const* qkv_bias;
    int32_t const* sequence_lengths;                  //
    int32_t const* context_lengths;                   // maybe not used now
    void const* alibi_slopes;                         // maybe not used now
    float const* rotary_embedding_inv_freq_cache;     // precomputed rotary inv freq
    int32_t const* spec_decoding_packed_mask;
    int const* spec_decoding_position_offsets;        // for position embedding.
    int const* spec_decoding_generation_lengths;      // variable input lengths.
    bool spec_decoding_is_generation_length_variable; // whether the generation lengths actually vary
    int32_t spec_decoding_max_generation_length;      // max possible input length
    int32_t const* mrope_position_deltas = nullptr;

    // almost copy from GPTAttentionPluginCommon.
    // maybe use one struct for parameters in GPTAttentionPluginCommon and share the same here.
    int32_t generation_input_length;
    int32_t num_q_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_size = 0;
    int unidirectional;
    float q_scaling = 0;
    int32_t rotary_embedding_dim = 0;
    float rotary_embedding_base = 0.0f;
    tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type;
    float rotary_embedding_scale;
    int rotary_embedding_max_positions;
    int rotary_vision_start;
    int rotary_vision_length;
    float2 const* rotary_cos_sin;
    tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled = false;
    bool remove_padding = false;
    tensorrt_llm::kernels::AttentionMaskType mask_type;
    // Paged KV cache parameters.
    bool paged_kv_cache;
    int tokens_per_block;
    int max_blocks_per_sequence;
    tensorrt_llm::common::QuantMode kv_cache_quant_mode;
    int tp_size = 1;
    int tp_rank = 0;
    bool qkv_bias_enabled;
    bool cross_attention;
    int max_distance = 0;
    bool multi_block_mode;
    bool multi_query_tokens = false;
    bool is_spec_dec_tree
        = true; // by default, XQA spec-dec expect tree-based draft token, only affective when multi_query_tokens = true

    float const* logn_scaling_ptr = nullptr; // for logn scaling in XQA

    int32_t total_num_input_tokens;          // total number of input tokens. may differ from batch_size due to medusa.
    bool is_fp8_output;
    float const* fp8_out_scale = nullptr; // fp8 output scale in case we need post-processing to convert output to fp8.
                                          // nullptr means no conversion.
    float const* fp4_out_sf_scale = nullptr; // SF scale for FP4 output.
    int32_t start_token_idx_sf = 0;          // The start token index in SF tensor.

    void* quant_q_buffer_ptr = nullptr;

    // for cross attention
    int32_t const* encoder_input_lengths = nullptr;

    // sparse attention parameters
    int32_t* sparse_attn_indices = nullptr;
    int32_t* sparse_attn_offsets = nullptr;
    bool use_sparse_attention = false;

    cudaStream_t stream = 0;

    std::string toString() const
    {
        std::stringstream ss;

        ss << "XQAParams ====================" << std::endl
           << "data_type: " << static_cast<int>(data_type) << std::endl
           << "kv_cache_data_type: " << static_cast<int>(kv_cache_data_type) << std::endl
           << "output: " << output << std::endl
           << "qkv: " << qkv << std::endl
           << "cache_indir: " << cache_indir << std::endl
           << "kv_scale_orig_quant: " << kv_scale_orig_quant << std::endl
           << "kv_scale_quant_orig: " << kv_scale_quant_orig << std::endl
           << "host_past_key_value_lengths: " << host_past_key_value_lengths << std::endl
           << "host_context_lengths: " << host_context_lengths << std::endl
           << "semaphores: " << semaphores << std::endl
           << "workspaces: " << workspaces << std::endl
           << "batch_size: " << batch_size << std::endl
           << "beam_width: " << beam_width << std::endl
           << "max_attention_window_size: " << max_attention_window_size << std::endl
           << "cyclic_attention_window_size: " << cyclic_attention_window_size << std::endl
           << "sink_token_length: " << sink_token_length << std::endl
           << "max_past_kv_length: " << max_past_kv_length << std::endl
           << "qkv_bias: " << qkv_bias << std::endl
           << "sequence_lengths: " << sequence_lengths << std::endl
           << "context_lengths: " << context_lengths << std::endl
           << "alibi_slopes: " << alibi_slopes << std::endl
           << "rotary_embedding_inv_freq_cache: " << rotary_embedding_inv_freq_cache << std::endl
           << "spec_decoding_packed_mask: " << spec_decoding_packed_mask << std::endl
           << "spec_decoding_position_offsets: " << spec_decoding_position_offsets << std::endl
           << "spec_decoding_generation_lengths: " << spec_decoding_generation_lengths << std::endl
           << "spec_decoding_is_generation_length_variable: "
           << (spec_decoding_is_generation_length_variable ? "true" : "false") << std::endl
           << "spec_decoding_max_generation_length: " << spec_decoding_max_generation_length << std::endl
           << "mrope_position_deltas: " << mrope_position_deltas << std::endl
           << "generation_input_length: " << generation_input_length << std::endl
           << "num_q_heads: " << num_q_heads << std::endl
           << "num_kv_heads: " << num_kv_heads << std::endl
           << "head_size: " << head_size << std::endl
           << "unidirectional: " << unidirectional << std::endl
           << "q_scaling: " << q_scaling << std::endl
           << "rotary_embedding_dim: " << rotary_embedding_dim << std::endl
           << "rotary_embedding_base: " << rotary_embedding_base << std::endl
           << "rotary_embedding_scale_type: " << static_cast<int>(rotary_embedding_scale_type) << " (enum value)"
           << std::endl
           << "rotary_embedding_scale: " << rotary_embedding_scale << std::endl
           << "rotary_embedding_max_positions: " << rotary_embedding_max_positions << std::endl
           << "rotary_vision_start: " << rotary_vision_start << std::endl
           << "rotary_vision_length: " << rotary_vision_length << std::endl
           << "rotary_cos_sin: " << rotary_cos_sin << std::endl
           << "position_embedding_type: " << static_cast<int>(position_embedding_type) << " (enum value)" << std::endl
           << "position_shift_enabled: " << (position_shift_enabled ? "true" : "false") << std::endl
           << "remove_padding: " << (remove_padding ? "true" : "false") << std::endl
           << "mask_type: " << static_cast<int>(mask_type) << " (enum value)" << std::endl
           << "paged_kv_cache: " << (paged_kv_cache ? "true" : "false") << std::endl
           << "tokens_per_block: " << tokens_per_block << std::endl
           << "max_blocks_per_sequence: " << max_blocks_per_sequence << std::endl
           << "tp_size: " << tp_size << std::endl
           << "tp_rank: " << tp_rank << std::endl
           << "qkv_bias_enabled: " << (qkv_bias_enabled ? "true" : "false") << std::endl
           << "cross_attention: " << (cross_attention ? "true" : "false") << std::endl
           << "max_distance: " << max_distance << std::endl
           << "multi_block_mode: " << (multi_block_mode ? "true" : "false") << std::endl
           << "multi_query_tokens: " << (multi_query_tokens ? "true" : "false") << std::endl
           << "logn_scaling_ptr :" << logn_scaling_ptr << std ::endl
           << "total_num_input_tokens :" << total_num_input_tokens << std ::endl
           << "is_fp8_output :" << (is_fp8_output ? "true" : "false") << std ::endl
           << "fp8_out_scale :" << fp8_out_scale << std ::endl
           << "encoder_input_lengths: " << encoder_input_lengths << std::endl
           << "sparse_attn_indices :" << sparse_attn_indices << std ::endl
           << "sparse_attn_offsets :" << sparse_attn_offsets << std ::endl
           << "use_sparse_attention :" << (use_sparse_attention ? "true" : "false") << std ::endl
           << "stream :" << stream;

        return ss.str();
    }

    bool isMLA() const
    {
        return head_size == 576 && num_q_heads == 128 && num_kv_heads == 1;
    }
};

} // namespace kernels
} // namespace tensorrt_llm
