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
    int timestep = 0;
    const void* qkv_bias;
    const int32_t* sequence_lengths; //
    const int32_t* context_lengths;  // maybe not used now
    const void* alibi_slopes;        // maybe not used now

    // almost copy from GPTAttentionPluginCommon.
    // maybe use one struct for parameters in GPTAttentionPluginCommon and share the same here.
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
    bool remove_padding = false;
    tensorrt_llm::kernels::AttentionMaskType mask_type;
    bool paged_kv_cache;
    int tokens_per_block;
    tensorrt_llm::common::QuantMode kv_cache_quant_mode;
    int tp_size = 1;
    int tp_rank = 0;
    bool qkv_bias_enabled;
    bool cross_attention;
    int max_distance = 0;
    bool multi_block_mode;
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
        if (xqaParams.data_type != DATA_TYPE_FP16)
            SUPPORT_RETURN_FALSE("data type");
        const int nbQHeads = xqaParams.num_q_heads;
        const int nbKVHeads = xqaParams.num_kv_heads;
        const int nbQHeadsPerKV = nbQHeads / nbKVHeads;
        if (nbQHeadsPerKV != 8 || (nbKVHeads != 1 && nbKVHeads != 2 && nbKVHeads != 4 && nbKVHeads != 8))
            SUPPORT_RETURN_FALSE("nbHeads");
        if (xqaParams.head_size != 128)
            SUPPORT_RETURN_FALSE("head_size");
        if (xqaParams.unidirectional != 1)
            SUPPORT_RETURN_FALSE("unidirectional");
        if (xqaParams.q_scaling != 1.0f)
            SUPPORT_RETURN_FALSE("q_scaling");
        if (xqaParams.rotary_embedding_dim != xqaParams.head_size)
            SUPPORT_RETURN_FALSE("rotary_embedding_dim");
        if (xqaParams.rotary_embedding_base != 10000.0f)
            SUPPORT_RETURN_FALSE("rotary_embedding_base");
        if (xqaParams.rotary_embedding_scale_type != tensorrt_llm::kernels::RotaryScalingType::kNONE)
            SUPPORT_RETURN_FALSE("rotary_embedding_scale_type");
        if (xqaParams.mask_type != tensorrt_llm::kernels::AttentionMaskType::CAUSAL)
            SUPPORT_RETURN_FALSE("mask_type");
        if (xqaParams.paged_kv_cache)
            SUPPORT_RETURN_FALSE("paged_kv_cache");
        if (xqaParams.qkv_bias_enabled)
            SUPPORT_RETURN_FALSE("qkv_bias_enabled");
        if (xqaParams.cross_attention)
            SUPPORT_RETURN_FALSE("cross_attention");

        if (xqaParams.host_past_key_value_lengths == nullptr)
            SUPPORT_RETURN_FALSE("host_past_key_value_lengths");
        if (xqaParams.beam_width != 1)
            SUPPORT_RETURN_FALSE("beam_width");
        if (xqaParams.cyclic_attention_window_size != xqaParams.max_attention_window_size)
            SUPPORT_RETURN_FALSE("cyclic_attention_window_size != max_attention_window_size");
        return shouldUseImpl(xqaParams);
    }

    size_t getWorkspaceSize();

    template <typename KVCacheBuffer>
    void dispatch(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
    {
        // TODO: Enable this when kernel supports KVBlockArray
        TLLM_CHECK_WITH_INFO((std::is_same<KVCacheBuffer, KVLinearBuffer>::value),
            "DecoderXQARunner.dispatch supports only KVLinearBuffer now.");
        sync_check_cuda_error();
        this->dispatchCacheBuffer(xqa_params, kv_cache_buffer, stream);
    }

private:
    void dispatchCacheBuffer(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream)
    {
        run(xqa_params, kv_linear_buffer, stream);
    }

    void dispatchCacheBuffer(const XQAParams& xqa_params, KVBlockArray& kv_block_array, const cudaStream_t& stream)
    {
        // TODO: Remove this when kernel supports KVBlockArray
        TLLM_CHECK_WITH_INFO(false, "DecoderXQARunner.dispatch doesn't support KVBlockArray now.");
    }

    bool shouldUseImpl(const XQAParams& xqaParams);
    void run(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream);

    // max number of CTAs for each KV head, multiple CTAs for one KV head is multi-block mode.
    // this number defines the maximum number when reaches both max_batch_size and max_beam_width.
    // If batch_size or beam_width doesn't reach maximum value, it is possible to have more CTAs per KV head than this
    // value.
    static constexpr int kMaxNbCtaPerKVHeadFactor = 4;

    static constexpr int kMaxBeamWidth = 4;

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
