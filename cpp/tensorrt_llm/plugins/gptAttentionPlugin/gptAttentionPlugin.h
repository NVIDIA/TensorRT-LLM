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
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include <cassert>
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

// inputs
//     0.  input_tensor [batch_size, seq_len, local_hidden_size + 2 * local_num_kv_heads * head_size] or
//                      [1, num_tokens, local_hidden_size + 2 * local_num_kv_heads * head_size] when
//                      enable_remove_input_padding
//     1.  sequence_length [batch_size]
//     2.  host_past_key_value_lengths [batch_size] (int32)
//     3.  context_lengths [batch_size]
//     4.  cache_indir [num_gen_requests, beam_width, memory_max_len] (required in beamsearch)
//     5.  host_request_types [batch_size] int32. 0: context; 1: generation: 2: none. When not in inflight-batching
//     mode,
//                      all elements must be identical.
//     6.  past_key_value_pool [batch_size, 2, local_num_kv_heads, max_seq_len, head_size] or
//         block_pointers [batch_size, 2, max_blocks_per_seq] if paged kv cache
//     7.  kv_cache_quantization_scale [1] (optional)
//     8.  kv_cache_dequantization_scale [1] (optional)
//     9.  alibi_slopes [num_heads] (optional for ALiBi position embedding)
//     10. host_context_lengths [batch_size] int32. (optional, required when remove_input_padding is true)
//     11. qkv_bias (optional) [local_hidden_size * 3]
//
// outputs
//     output_tensor [batch_size, seq_len, local_hidden_size]
//     present_key_value_pool (optional if not paged kv cache) [batch_size, 2, local_num_kv_heads, max_seq_len,
//     head_size]

class GPTAttentionPlugin : public GPTAttentionPluginCommon
{
public:
    GPTAttentionPlugin(int num_heads, int num_kv_heads, int head_size, int unidirectional, float q_scaling,
        tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim, // for RoPE. 0 for non-RoPE
        float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, int rotary_embedding_max_positions, int tp_size, int tp_rank, // for ALiBi
        tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, int kv_cache_quant_mode,
        bool remove_input_padding, tensorrt_llm::kernels::AttentionMaskType mask_type, bool paged_kv_cache,
        int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled,
        bool cross_attention = false, int max_distance = 0);

    GPTAttentionPlugin(const void* data, size_t length);

    ~GPTAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    template <typename T, typename KVCacheBuffer>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    template <typename T>
    int enqueueDispatchKVCacheType(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    GPTAttentionPlugin* clone() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

private:
    template <typename T, typename KVCacheBuffer>
    int enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg, int32_t localNbTokens,
        const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    using IndexType = std::int32_t;

    IndexType getInputTensorIdx() const
    {
        return 0;
    }

    IndexType getSequenceLengthIdx() const
    {
        return 1;
    }

    IndexType getHostPastKeyValueLengthsIdx() const
    {
        return 2;
    }

    IndexType getContextLengthsIdx() const
    {
        return 3;
    }

    IndexType getCacheIndirIdx() const
    {
        return 4;
    }

    IndexType getRequestTypesIdx() const
    {
        return 5;
    }

    IndexType getKVCacheBlockPointersIdx() const
    {
        // NOTE We either provide this tensor when mPagedKVCache is true or PastKeyValue otherwise
        return 6;
    }

    IndexType getPastKeyValueIdx() const
    {
        // NOTE We either provide this tensor when mPagedKVCache is false or KVCacheBlockPointers otherwise
        return 6;
    }

    IndexType getKVCacheQuantizationScaleIdx() const
    {
        return 7;
    }

    IndexType getKVCacheDequantizationScaleIdx() const
    {
        return 8;
    }

    IndexType getAlibiSlopesIdx() const
    {
        return (mKVCacheQuantMode.hasKvCacheQuant() ? 9 : 7);
    }

    IndexType getRelativeAttentionBiasIdx() const
    {
        return getAlibiSlopesIdx() + (isALiBi() ? 1 : 0);
    }

    IndexType getCrossQKVIdx() const
    {
        return getRelativeAttentionBiasIdx() + (isRelativePosition() ? 1 : 0);
    }

    IndexType getCrossQKVLengthIdx() const
    {
        return getCrossQKVIdx() + 1;
    }

    IndexType getEncoderInputLengthsIdx() const
    {
        return getCrossQKVLengthIdx() + 1;
    }

    IndexType getHostContextLengthsIdx() const
    {
        TLLM_CHECK(mRemovePadding);
        return getCrossQKVIdx() + (isCrossAttention() ? 3 : 0);
    }

    IndexType getQKVBiasTensorIdx() const
    {
        TLLM_CHECK(mQKVBiasEnabled);
        return (mKVCacheQuantMode.hasKvCacheQuant() ? 9 : 7) + (isALiBi() ? 1 : 0) + (mRemovePadding ? 1 : 0);
    }
};

class GPTAttentionPluginCreator : public GPTAttentionPluginCreatorCommon
{
public:
    GPTAttentionPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

} // namespace tensorrt_llm::plugins
