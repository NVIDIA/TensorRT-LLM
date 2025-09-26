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

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::kernels
{
class DecoderXQARunnerResource;
}

namespace tensorrt_llm::plugins
{

class GPTAttentionPluginCommon : public BasePlugin, public tensorrt_llm::common::op::AttentionOp
{
public:
    GPTAttentionPluginCommon() = delete;

    GPTAttentionPluginCommon(int layer_idx, int num_heads, int vision_start, int vision_length, int num_kv_heads,
        int num_kv_heads_origin, int head_size, int unidirectional, float q_scaling, float attn_logit_softcapping_scale,
        tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim, // for RoPE. Use 0 for non-RoPE
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
        bool spec_decoding_is_generation_length_variable = false, int32_t spec_decoding_max_generation_length = 1,
        bool is_mla_enabled = false, int q_lora_rank = 0, int kv_lora_rank = 0, int qk_nope_head_dim = 0,
        int qk_rope_head_dim = 0, int v_head_dim = 0, bool fuse_fp4_quant = false, bool skip_attn = false,
        int cp_size = 1, int cp_rank = 0, std::set<int32_t> cp_group = {});

    GPTAttentionPluginCommon(void const* data, size_t length);

    ~GPTAttentionPluginCommon() override = default;

    template <typename T>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    //! This is called on every trt Engine creation
    int initialize() noexcept override;
    //! This is called on every trt Engine destroy
    void terminate() noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    template <typename T>
    T* cloneImpl() const noexcept;

    //! This is called on evert trt Engine or ExecutionContext destroy.
    //! None-cloned plugins will call terminate and then call destroy, while the cloned plugins will call destroy only
    //! So plugin should put the resource release inside destroy.
    void destroy() noexcept override;

    size_t getCommonSerializationSize() const noexcept;
    void serializeCommon(void* buffer) const noexcept;

protected:
    std::string const mLayerName;

private:
    std::shared_ptr<tensorrt_llm::kernels::DecoderXQARunnerResource> mResource;
};

class GPTAttentionPluginCreatorCommon : public BaseCreator
{
public:
    GPTAttentionPluginCreatorCommon();

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    template <typename T>
    T* deserializePluginImpl(char const* name, void const* serialData, size_t serialLength) noexcept;

protected:
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};

} // namespace tensorrt_llm::plugins
