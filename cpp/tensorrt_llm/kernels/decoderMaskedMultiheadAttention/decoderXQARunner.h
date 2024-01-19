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

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, typename KVCacheBuffer>
struct XQADispatchHelper
{
    static constexpr bool CanSupport = false;
};

template <>
struct XQADispatchHelper<__half, KVLinearBuffer>
{
    static constexpr bool CanSupport = true;
};

template <>
struct XQADispatchHelper<__half, KVBlockArray>
{
    static constexpr bool CanSupport = true;
};

#ifdef ENABLE_BF16
template <>
struct XQADispatchHelper<__nv_bfloat16, KVLinearBuffer>
{
    static constexpr bool CanSupport = true;
};

template <>
struct XQADispatchHelper<__nv_bfloat16, KVBlockArray>
{
    static constexpr bool CanSupport = true;
};
#endif

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

#define SUPPORT_RETURN_FALSE(X)                                                                                        \
    {                                                                                                                  \
        return false;                                                                                                  \
    }

class DecoderXQARunner
{
public:
    DecoderXQARunner(
        const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode);
    ~DecoderXQARunner();

    template <typename T>
    bool shouldUse(const XQAParams& xqaParams)
    {
        if (!(xqaParams.data_type == DATA_TYPE_FP16 || xqaParams.data_type == DATA_TYPE_BF16))
        {
            SUPPORT_RETURN_FALSE("data type");
        }
        if (xqaParams.head_size != 128)
        {
            SUPPORT_RETURN_FALSE("head_size");
        }
        if (xqaParams.unidirectional != 1)
        {
            SUPPORT_RETURN_FALSE("unidirectional");
        }
        if (xqaParams.q_scaling != 1.0f)
        {
            SUPPORT_RETURN_FALSE("q_scaling");
        }
        if (xqaParams.mask_type != tensorrt_llm::kernels::AttentionMaskType::CAUSAL)
        {
            SUPPORT_RETURN_FALSE("mask_type");
        }
        if (xqaParams.cross_attention)
        {
            SUPPORT_RETURN_FALSE("cross_attention");
        }
        // Only support 64/128 tokens per block.
        if (xqaParams.paged_kv_cache && xqaParams.tokens_per_block != 64 && xqaParams.tokens_per_block != 128)
        {
            SUPPORT_RETURN_FALSE("paged_kv_cache");
        }
        if (xqaParams.host_past_key_value_lengths == nullptr)
        {
            SUPPORT_RETURN_FALSE("host_past_key_value_lengths");
        }
        if (xqaParams.beam_width != 1)
        {
            SUPPORT_RETURN_FALSE("beam_width");
        }
        if (xqaParams.cyclic_attention_window_size != xqaParams.max_attention_window_size)
        {
            SUPPORT_RETURN_FALSE("cyclic_attention_window_size != max_attention_window_size");
        }
        if (xqaParams.position_shift_enabled || xqaParams.sink_token_length > 0)
        {
            SUPPORT_RETURN_FALSE("streaming-llm");
        }

        // OPTIMIZE: For the standard generation-phase MHA, there are still extra limitations.
        // NOTE: Medusa mode = Multi_query_tokens > 1.
        const int nbQHeads = xqaParams.num_q_heads;
        const int nbKVHeads = xqaParams.num_kv_heads;
        const int nbQHeadsPerKV = nbQHeads / nbKVHeads;
        if (!xqaParams.multi_query_tokens)
        {
            if (nbQHeadsPerKV != 8 || (nbKVHeads != 1 && nbKVHeads != 2 && nbKVHeads != 4 && nbKVHeads != 8))
            {
                SUPPORT_RETURN_FALSE("nbHeads");
            }
        }
        else
        {
            // Number of Q heads Per KV needs to be power of 2 or 1.
            if (!(nbQHeadsPerKV % 2 == 0 || nbQHeadsPerKV == 1))
            {
                SUPPORT_RETURN_FALSE("nbHeads");
            }

            // TODO: add fp8/int8 kv cache kernels.
            if (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 || xqaParams.kv_cache_data_type == DATA_TYPE_INT8)
            {
                SUPPORT_RETURN_FALSE("KV cache data type");
            }
        }
        return shouldUseImpl(xqaParams);
    }

    size_t getWorkspaceSize(int max_batch_beam_size);

    template <typename KVCacheBuffer>
    void dispatch(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
    {
        sync_check_cuda_error();
        this->run(xqa_params, kv_cache_buffer, stream);
    }

private:
    bool shouldUseImpl(const XQAParams& xqaParams);

    template <typename KVCacheBuffer>
    void run(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream);

    static constexpr int kMaxBeamWidth = 4;

    // Cache the grid_size and block_size that gives the highest occupancy for
    //  invokeApplyBiasRopeUpdateKVCache.
    int2 mLaunchGridBlockCache = make_int2(0, 0);

    class xqaImpl;
    std::unique_ptr<xqaImpl> pimpl;

    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    bool mMultiBlockMode;
    int mMultiProcessorCount;
};

} // namespace kernels
} // namespace tensorrt_llm
