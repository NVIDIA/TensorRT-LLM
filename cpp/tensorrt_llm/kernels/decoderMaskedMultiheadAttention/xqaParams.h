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
    const void* qkv = nullptr;
    const int32_t* cache_indir = nullptr;
    const float* kv_scale_orig_quant = nullptr;
    const float* kv_scale_quant_orig = nullptr;
    const int32_t* host_past_key_value_lengths = nullptr;
    const int32_t* host_context_lengths = nullptr;
    void* workspaces = nullptr;
    uint32_t batch_size = 0;
    int32_t beam_width = 0;
    int32_t max_attention_window_size = 0;
    int32_t cyclic_attention_window_size = 0;
    int32_t sink_token_length = 0;
    int timestep = 0;
    const void* qkv_bias;
    const int32_t* sequence_lengths;    //
    const int32_t* context_lengths;     // maybe not used now
    const void* alibi_slopes;           // maybe not used now
    const int32_t* medusa_packed_mask;
    const int* medusa_position_offsets; // rotary embedding.

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
};

} // namespace kernels
} // namespace tensorrt_llm
