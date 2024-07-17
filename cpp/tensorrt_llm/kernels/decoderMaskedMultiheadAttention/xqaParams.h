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
    void* output = nullptr;
    void const* qkv = nullptr;
    int32_t const* cache_indir = nullptr;
    float const* kv_scale_orig_quant = nullptr;
    float const* kv_scale_quant_orig = nullptr;
    int32_t const* host_past_key_value_lengths = nullptr;
    int32_t const* host_context_lengths = nullptr;
    int32_t* semaphores = nullptr;
    void* workspaces = nullptr;
    uint32_t batch_size = 0;
    int32_t beam_width = 0;
    int32_t max_attention_window_size = 0;
    int32_t cyclic_attention_window_size = 0;
    int32_t sink_token_length = 0;
    int timestep = 0;
    void const* qkv_bias;
    int32_t const* sequence_lengths;             //
    int32_t const* context_lengths;              // maybe not used now
    void const* alibi_slopes;                    // maybe not used now
    int32_t const* spec_decoding_packed_mask;
    int const* spec_decoding_position_offsets;   // rotary embedding.
    int const* spec_decoding_generation_lengths; // variable input lengths.

    // almost copy from GPTAttentionPluginCommon.
    // maybe use one struct for parameters in GPTAttentionPluginCommon and share the same here.
    int32_t generation_input_length;
    int32_t layer_idx = 0;
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

    int32_t total_num_input_tokens;       // total number of input tokens. may differ from batch_size due to medusa.
    float const* fp8_out_scale = nullptr; // fp8 output scale in case we need post-processing to convert output to fp8.
                                          // nullptr means no conversion.
};

} // namespace kernels
} // namespace tensorrt_llm
